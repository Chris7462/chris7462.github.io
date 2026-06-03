---
sidebar_position: 3
title: Screen Configuration
description: Personal GNU Screen configuration for Yi-Chen Zhang
---

# Screen Configuration

For personal use, place the configuration in `~/.screenrc`. For system-wide use (all users), append it to `/etc/screenrc`:

```bash
caption always "%{= wk} %{= KY} [%n]%t @ %H %{-} %= %{= KR} %l %{-} | %{= KG}%Y-%m-%d %{-} "
hardstatus alwayslastline " %-Lw%{= Bw}%n%f %t%{-}%+Lw %=| %0c:%s "
```

For system-wide use, add the lines to `/etc/screenrc`:

```bash
sudo vim /etc/screenrc
```

The `caption` line adds a status bar to each window showing the window number `[%n]`, window title `%t`, and hostname `%H`. The `hardstatus` line adds a bottom bar showing all open windows with the current one highlighted, plus the current time.
