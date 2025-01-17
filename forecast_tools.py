import math
import re
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_FORECAST_DESCRIPTION = (
    "forecast(problem: str, context: Optional[list[str]]) -> Dict[str, List[float]]:\n"
    " - Generates forecasts based on the provided problem description.\n"
    ' - `problem` should describe the forecasting task (e.g. "forecast next 3 months based on [1,2,3,4,5]")\n'
    " - You cannot forecast multiple series in one call. For instance, forecasting two different datasets needs separate calls\n"
    " - Minimize the number of `forecast` actions as much as possible\n"
    " - You can optionally provide a list of strings as `context` to help with additional data or parameters\n"
    " - When asking about forecasts, specify the time units. For instance, 'forecast next 3 months' or 'predict next 5 days'\n"
)

_SYSTEM_PROMPT = """Translate a forecasting problem into an expression that can be executed. Use the output of running this code to answer the question.

Question: {problem}
```text
{code expression that forecasts the result}
```
...processing forecast...
```output
{Output of running the code}
```
Answer: {forecast result}

Begin.
"""

_ADDITIONAL_CONTEXT_PROMPT = """Additional context for the forecast:
{context}

Use this information to adjust the forecast parameters or interpretation as needed."""

class ExecuteCode(BaseModel):
    """The input for the forecasting function."""
    
    reasoning: str = Field(
        ...,
        description="The reasoning behind the forecasting expression, including how context is included.",
    )
    
    code: str = Field(
        ...,
        description="The code expression to execute the forecast.",
    )

def _execute_forecast(expression: str) -> str:
    try:
        # Parse the expression to extract parameters
        local_dict = {}
        exec(expression, {"__builtins__": {}}, local_dict)
        
        # Generate forecast using the parameters
        result = _generate_forecast(
            data=local_dict['data'],
            periods=local_dict['periods'],
            method=local_dict.get('method', 'holt_winters')
        )
        return str(result)
    except Exception as e:
        raise ValueError(
            f'Failed to generate forecast for "{expression}". Error: {repr(e)}.'
            " Please try again with valid parameters"
        )

def get_forecast_tool(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    
    # Simplify the chain to match math_tools.py
    chain = prompt | llm.with_structured_output(ExecuteCode, method="function_calling")
    
    def generate_forecast(
        problem: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"problem": problem}
        
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
                chain_input["context"] = [SystemMessage(content=context_str)]
            
        code_model = chain.invoke(chain_input, config)
        try:
            return _execute_forecast(code_model.code)
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="forecast",
        func=generate_forecast,
        description=_FORECAST_DESCRIPTION,
    )

def _generate_forecast(data: List[float], periods: int, method: str = "holt_winters") -> Dict[str, List[float]]:
    try:
        series = pd.Series(data)
        
        if method == "holt_winters":
            # Holt-Winters method for time series with trend and seasonality
            model = ExponentialSmoothing(
                series,
                seasonal_periods=min(len(series) // 2, 12),  # Auto-detect seasonality
                trend='add',
                seasonal='add'
            ).fit()
            
            forecast = model.forecast(periods)
            # Generate confidence intervals (simple implementation)
            confidence_intervals = [
                [f - 1.96 * model.sse, f + 1.96 * model.sse] for f in forecast
            ]
            
        elif method == "moving_average":
            # Simple moving average
            window = min(len(series) // 3, 12)
            ma = series.rolling(window=window).mean()
            last_ma = ma.iloc[-1]
            forecast = [last_ma] * periods
            std = series.std()
            confidence_intervals = [
                [f - 1.96 * std, f + 1.96 * std] for f in forecast
            ]
            
        return {
            "forecast": forecast.tolist() if isinstance(forecast, pd.Series) else forecast,
            "confidence_intervals": confidence_intervals
        }
        
    except Exception as e:
        raise ValueError(f"Failed to generate forecast. Error: {repr(e)}")
      