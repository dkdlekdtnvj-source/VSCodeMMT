# UTF-8 Encoding Guide

The project assumes UTF-8 everywhere.  Follow the steps below to keep your
environment aligned, especially on Windows where the legacy ANSI code page can
surface.

## Windows

1. **Set persistent Python variables** (already executed by the automation, but
   safe to repeat):

   ```powershell
   setx PYTHONUTF8 1
   setx PYTHONIOENCODING utf-8
   ```

   Restart shells to apply the changes.

2. **Default PowerShell to UTF-8** by adding the following snippet to your
   PowerShell profile (run `notepad $PROFILE` to edit):

   ```powershell
   chcp 65001 > $null
   $env:PYTHONUTF8 = "1"
   $env:PYTHONIOENCODING = "utf-8"
   ```

3. **Optional:** enable *"Beta: Use Unicode UTF-8 for worldwide language support"*
   under *Region -> Administrative -> Change system locale...* for a system-wide
   switch. This requires a reboot.

## macOS / Linux

Most modern distributions already run shells in UTF-8.  Ensure your locale is
set properly:

```bash
locale
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
```

Persist the exports in your preferred shell profile (e.g. `~/.bashrc` or
`~/.zshrc`).

## Editor Settings

- `.editorconfig` at the repository root enforces UTF-8 for all text files.
- Configure your IDE/Editor to save files as UTF-8 with LF line endings.

## Python Safeguards

- `sitecustomize.py` reconfigures `stdin`, `stdout`, and `stderr` to UTF-8
  automatically when Python starts with the repository on `PYTHONPATH`.
- When opening files manually, always specify `encoding="utf-8"`:

  ```python
  with open(path, "w", encoding="utf-8") as fh:
      fh.write("...")
  ```

Adhering to these steps prevents mojibake when collaborating across different
platforms.
