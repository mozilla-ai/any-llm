# Contributing to mozilla.ai any-llm

Thank you for your interest in contributing to this repository! This project supports the mozilla.ai goal of empowering developers to integrate AI capabilities into their projects using open-source tools and models.

We welcome all kinds of contributions, from improving customization, to extending capabilities, to fixing bugs. Whether you’re an experienced developer or just starting out, your support is highly appreciated.

---

## **How to Contribute**

### **Customize for your use-case or Extend It** 🔧
- Fork this repo and customize it for your own use-case or even extend its capabilities.
- We'd love to see what you've built!

### **Browse Existing Issues** 🔍
- Check the Issues page to see if there are any tasks you'd like to tackle.
- Look for issues labeled **`good first issue`** if you're new to the project—they're a great place to start.

### **Report Issues** 🐛
- Found a bug? Open a Bug Report by clicking on 'New Issue'
- Provide as much detail as possible, including the steps to reproduce the issue and Expected vs. actual behavior

### **Suggest Features** 🚀
- Have an idea for improving the project? Open a Feature Request by clicking on 'New Issue'
- Share why the feature is important and any alternative solutions you’ve considered.

### **Submit Pull Requests** 💻
- Fork the repository and create a new branch for your changes.
- Follow the [Guidelines for Contributions](#guidelines-for-contributions)
- Ensure your branch is up-to-date with the main branch before submitting the PR.
- Please follow the PR template, adding as much detail as possible, including how to test the changes

---

### **Guidelines for Contributions**

**Install**

We recommend to use [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv venv
source .venv/bin/activate
uv sync --dev --extra all
```

**Linting**

Ensure all the checks pass:

```bash
pre-commit run --all-files
```

**Testing**

Test changes locally to ensure functionality.

```bash
pytest -v tests
```

**Documentation**

Update docs for changes to functionality and maintain consistency with existing docs.

```bash
mkdocs serve
```
