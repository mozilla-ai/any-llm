---
name: Client Generation
about: Generate a language implementation of any-llm
title: "[Generation]"
labels: generation
assignees: ''

---

Generate a {language} library implementation based on the given blueprint. 

You must:

- Carefully read the contents of blueprint
- Implement errors.
- Implement types.
- Implement the AnyLLM base class.
- Implement the OpenAI provider.
- Test the implementation against the acceptance tests.

Put the generated code at {language_stub}/src. 

Do not check other folders. If there is any ambiguous or missing information, add a note to clarifications.md but progress as much as you can by making assumptions.
