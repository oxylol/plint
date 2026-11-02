# Editing programs

`edit_program` changes an existing program in place: a list of exact `old` → `new` text replacements, applied in order, all or nothing, after which the simulation runs and you get the same report as from `write_program`. Use it for every change to a program that already exists: a feed, a depth, one more pass, a moved hole, a fixed line, a new tool section. Use `write_program` only for a new program, a new operation, or when more than about a third of the lines change.

- Each `old` must match the program exactly once. Copy it from the `<program>` block as it stands, spaces included; one or two more lines around the change make it unique. `new` replaces it; an empty `new` deletes it. An edit that does not match once is refused with the closest lines, and nothing is applied: correct that edit and call again.
- The number of tokens used to edit files is best minimized, all else being equal. Therefore, when it will not affect the end result, try to surgically edit a file rather than rewrite the entire thing.
- The report after an edit covers the whole program, as always; read it the same way, and fix with another edit.
