#!/usr/bin/env zsh
tensorboard --logdir=runs --samples_per_plugin images=100

tensorboard --logdir=scripts/runs --samples_per_plugin images=100

tensorboard --logdir=src/pyroNMF/models/runs --samples_per_plugin images=100