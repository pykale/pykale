---
name: "Action failed"
about: Raised when a GitHub action fails
title: '[Test action failed] {{ date | date('dddd, MMMM Do') }}'
labels: action failed
assignees: ''

---

A test in this context has failed: {{ tools.context.action }}.

For testing only, please ignore.
