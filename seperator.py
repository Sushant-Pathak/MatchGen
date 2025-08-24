import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import Optional
from typing import Optional
from pydantic import BaseModel, Field


# Definition of Pydantic Schema 
class Experience(BaseModel):
    company: str = Field(..., description="Name of the company")
    duration: str = Field(..., description="Duration of employment in years and months")
    skilss: Optional[list[str]]=Field(description="Skills used to devlop the workspace in the company")
    impact_works: Optional[list[str]]=Field(description="Work done or specific projects or expertise delivered to the company")

class ResumeInfo(BaseModel):
    experience_no_year: list[Experience] = Field(
        default_factory=list,
        description="Work experience with company and duration in year and month (e.g. 1 year if march 2022 - march 2023)"
    )
    skills: list[str] = Field(
        default_factory=list,
        description="List of candidate's skills"
    )
    education: list[str] = Field(
        default_factory=list,
        description="Educational qualifications"
    )
    projects: list[str] = Field(
        default_factory=list,
        description="List of projects done"
    )
    project_skills: list[str] = Field(
        default_factory=list,
        description="Skills specifically used in projects"
    )
    achievements: Optional[list[str]] = Field(
        default=None,
        description="Certifications, positions of responsibility, or coding achievements (e.g., LeetCode, Codeforces, etc.)"
    )


# Load API key and setup LLM once
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Parser and prompt templateto provide to the llm
parser = PydanticOutputParser(pydantic_object=ResumeInfo)

prompt = PromptTemplate(
    template=(
        "You are an expert resume parser. Extract structured details from this resume text.\n\n"
        "Resume Text:\n{resume_text}\n\n"
        "Return ONLY valid JSON matching this format:\n{format_instructions}"
    ),
    input_variables=["resume_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


#  Function to parse resume text
def parse_resume_text(resume_text: str) -> ResumeInfo:
    # Takes raw resume text and returns structured ResumeInfo (skills, education, projects, etc.)
    input_prompt = prompt.format_prompt(resume_text=resume_text)
    output = llm.invoke(input_prompt.to_string())
    return parser.parse(output.content)