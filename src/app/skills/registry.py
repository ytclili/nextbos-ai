from pathlib import Path

from app.skills.loader import LoadedSkill, SkillLoader


class SkillRegistry:
    def __init__(self, root: str | Path):
        self.loader = SkillLoader(root)

    def load(self, name: str) -> LoadedSkill | None:
        return self.loader.load(name)
