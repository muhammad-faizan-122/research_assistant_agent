import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from google.api_core.exceptions import ResourceExhausted


def safe_invoke_llm(
    llm: ChatGoogleGenerativeAI,
    msg: list[HumanMessage],
    total_retries: int = 10,
    delay_sec: int = 10,
) -> BaseMessage:
    """
    Wraps the LLM invoke call with custom retry logic for ResourceExhausted errors.
    """
    attempt = 0
    last_exception = None
    while attempt <= total_retries:
        try:
            if attempt > 0:
                print(
                    f"Rate limit hit. Sleeping {delay_sec}s (Attempt {attempt}/{total_retries})..."
                )
                time.sleep(delay_sec + attempt * 2)
            return llm.invoke(msg)

        except ResourceExhausted as e:
            last_exception = e
            attempt += 1
        except Exception as e:
            raise e
    print("All retries exhausted.")
    if last_exception:
        raise last_exception
    else:
        raise Exception("Unknown error occurred during retry loop")
