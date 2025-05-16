---
title: Abraxus
tags:
  - abraxus
  - obsidian
  - plugins
source: Abraxus
---
[[Abraxus]] uses #obsidian and a rich #plugins ecosystem as well as Windows Sandbox, WSL, WindowsTerminal, and VScode to create a dev-env, a veritable operating-system.

```dataview
TABLE file.name AS Hub, length(rows) AS "Note Count"
FROM [[kb/Abraxus]]
FLATTEN file.outlinks AS hubLink
GROUP BY hubLink AS Hub
SORT Hub.file.name ASC
```
```dataview
TABLE file.name, length(rows) AS "Note Count"
FROM [[kb/Abraxus]]
FLATTEN file.outlinks
```

`TABLE file.name AS Hub, linkCount AS "Note Count"`: Creates a table with the hub's name and the count of notes linking to it.
`FLATTEN file.outlinks AS hubLink`: Gets the linked notes.
`WHERE contains(hubLink.file.folder, "Hub") OR contains(hubLink.file.folder, "Meta")`: Filters for your central hubs.
`GROUP BY hubLink AS Hub`: Groups the results by the linked hub.
`SORT Hub.file.name ASC`: Sorts the hubs alphabetically.
`linkCount = length(rows)` (Implicit): Dataview automatically counts the number of rows in each group, which represents the number of notes linking to that hub.
