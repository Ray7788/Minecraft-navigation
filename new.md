---
title: 21-12-2023 updates
date: 2023-12-21 12:11:07
tags: 
---
UPDATES
======
Initialise Dockerfile
------
* Make a new Dockerfile for the new project. Based on Minedojo and MineRL environment, remake new environment  and Intergrate existing dependencies(avoid conflict dependencies)
  * This Dockerfile installs a more extensive set of Xfce desktop environment components, including additional plugins and tools, much convinent and smaller than the traditional one.
  * installs noVNC and X11VNC for web-based remote desktop access.\
  * installs Miniconda with Python 3.9 and sets up the environment accordingly.
  * creates a user named "user" with a specified password (${PASSWD}) and adds it to various groups, including "sudo." It also configures passwordless sudo for this user.
  * includes an EXPOSE 8080 directive to indicate that the container will listen on port 8080.
  * installs the "MineDojo" Python package using pip3.
  * Still use MineRl v0.4.4, the latest version v1.0.2 includes VPT, which is useful for this new project but it has installation errors(many users have the same problem, see https://github.com/minerllabs/minerl/issues), but official developers from MineRL said Minedojo  has similar environment with mineRL v0.4.4, maybe it's a good choice.
