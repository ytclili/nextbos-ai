from pathlib import Path


class LoadedSkill:
    def __init__(self, name: str, description: str, content: str, path: Path):
        self.name = name
        self.description = description
        self.content = content
        self.path = path


class SkillLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, name: str) -> LoadedSkill | None:
        if not name or Path(name).name != name:
            return None
        path = self.root / name / "SKILL.md"
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8")
        description = ""
        for line in content.splitlines():
            if line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip()
                break
        return LoadedSkill(name=name, description=description, content=content, path=path)
