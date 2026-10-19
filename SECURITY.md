# Security Policy

## Reporting a vulnerability

Please report security issues privately, **not** as a public GitHub issue. Use
GitHub's [private vulnerability reporting](https://github.com/bnelair/mef3io-coregistration/security/advisories/new)
(Security → Report a vulnerability), or contact the maintainers directly.

Useful details: affected version, platform, a minimal reproduction, and the
impact you observed. We aim to acknowledge reports within a week.

## Scope

mef3io-coregistration reads MEF 3.0 recordings through
[mef3io](https://github.com/bnelair/mef3io) and reads coregistration results
and recording mappings from JSON files that may come from someone else. In
scope:

- a crafted result or mapping file that makes the package execute code, write
  outside the paths it was given, or exhaust memory or disk;
- a result applied to a recording it was not computed on without the error
  the time-span check promises (for example A and B swapped silently);
- anything that writes signal, patient details or passwords to a place the
  user did not ask for (logs, temporary files left behind, review reports).

Parsing of the MEF 3.0 files themselves is mef3io's; report issues there to
[mef3io's security policy](https://github.com/bnelair/mef3io/blob/main/SECURITY.md).

A result file holds no signal, but it does hold the recordings' paths, start
and end times and content hashes; treat it as you would the recordings'
metadata.

## Supported versions

Fixes land on the latest release line. Please upgrade to the newest version
before reporting.
