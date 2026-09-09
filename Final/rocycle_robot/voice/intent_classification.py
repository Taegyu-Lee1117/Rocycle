"""Voice intent classification for the recycling-sort cobot.

Adapted from the "공구(tool)" reference pipeline
(~/Downloads/corecode/corecode/VoiceProcessing/keyword_extraction.py), which
extracted a NOUN (tool name) + destination. Our utterances are different in
kind -- they select a COMMAND INTENT, not an object -- so this is a new
prompt/parser, not a wording tweak of the old one (설계문서 v31 4-1절).

Intents (설계문서 v31 4-2절 발화 목록):
    start            "분리수거 시작해줘"
    pause            "잠깐 멈춰"
    resume           "다시 시작해"
    stop             "그만하자"                (종료 보고 트리거)
    speed_inquiry    "천천히 해줘"              (고정 속도 안내 응답)
    handoff_ack      "받아 갈게요" 등           (핸드오버 응답)
    unknown          위 어디에도 안 맞음
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

VALID_INTENTS = (
    "start",
    "pause",
    "resume",
    "stop",
    "speed_inquiry",
    "handoff_ack",
    "unknown",
)

_PROMPT = """
당신은 컨베이어 재활용 분류 로봇에게 내려진 음성 명령의 "의도"를
아래 목록 중 하나로 분류해야 합니다. 목록에 없는 의도는 unknown으로
분류하세요.

<의도 목록>
- start: 분리수거 작업을 시작하라는 명령 (예: "분리수거 시작해줘", "시작하자")
- pause: 작업을 일시 정지하라는 명령 (예: "잠깐 멈춰", "정지")
- resume: 정지했던 작업을 다시 시작하라는 명령 (예: "다시 시작해", "계속해")
- stop: 작업을 완전히 종료하라는 명령 (예: "그만하자", "종료해줘", "끝내자")
- speed_inquiry: 속도를 바꿔달라는 요청 (예: "천천히 해줘", "속도 좀 줄여줘")
- handoff_ack: 로봇이 내미는 물건을 받겠다는 응답 (예: "받아 갈게요", "네 받을게요")
- unknown: 위 어디에도 명확히 속하지 않음

<출력 형식>
- 의도 하나만, 다른 말 없이 정확히 그 단어만 출력하세요.
  (예: start / pause / resume / stop / speed_inquiry / handoff_ack / unknown)

<사용자 입력>
"{user_input}"
"""


@dataclass(frozen=True)
class IntentResult:
    intent: str
    raw_response: str


class IntentClassifier:
    def __init__(self, openai_api_key: str | None = None) -> None:
        if openai_api_key is None:
            load_dotenv(dotenv_path=".env")
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (env var or .env) -- cannot call the "
                "classification model. See 설계문서 v31 작업 B."
            )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=_PROMPT
        )
        self.chain = self.prompt_template | self.llm

    def classify(self, utterance: str) -> IntentResult:
        response = self.chain.invoke({"user_input": utterance})
        raw = response.content.strip()
        intent = raw.lower()
        if intent not in VALID_INTENTS:
            intent = "unknown"
        return IntentResult(intent=intent, raw_response=raw)
