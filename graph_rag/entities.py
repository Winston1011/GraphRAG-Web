from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Entities(BaseModel):
    """
    从文本中识别出指定-有关实体的信息
    """

    names: List[str] = Field(
        description=
            "文本中出现的所有人物、组织或商业实体的名称",
    )