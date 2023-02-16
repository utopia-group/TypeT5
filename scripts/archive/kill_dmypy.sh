#!/bin/sh

ps aux | grep -i dmypy | awk '{print $2}' | xargs kill
