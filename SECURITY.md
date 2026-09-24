# Security

SignScribe runs entirely locally and makes no network requests. Things worth knowing:

* Trained models and recorded samples are plain NumPy `.npz` archives loaded with `allow_pickle=False`, so
  loading them cannot execute code.
* The optional "type into other applications" feature (needs the extra `pynput` package) sends keystrokes to
  whichever window has focus. It is off by default and only active when you tick it.
* Camera frames are processed in memory and never written to disk; only 21-point landmark samples you
  explicitly record are stored.

To report a vulnerability, please open a private security advisory on GitHub (Security tab, "Report a
vulnerability") rather than a public issue.
