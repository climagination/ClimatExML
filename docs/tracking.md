---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Comet

[Comet](https://comet.com/) is a tool that allows you to track your machine learning experiments in real time. It is fairly straightforward to get started. This documentations provides only basic instructions for getting set up with Comet: users should refer to the [Comet docs](https://www.comet.com/docs/v2/) for detailed functionality and configuration.
Note that previously, ClimatEx used MLFLOW for experiment tracking. The instructions for configuring MLFLOW can be found in the documentation `configuring.md`.

# Setting Up

First, the user must sign up for Comet [here](https://www.comet.com/signup). As of the writing of this documentation, signing up and using Comet is completely free, though there are paid plans that provide more advanced functionalities if desired. After signing up, the user will be provided with an API (Application Programming Interface) key, which will serve as a form of user authentication. The API key can be found by going to "Account Settings", then selecting "API Keys". It is important that the user keeps their API key secure.

In your training environment, install the `comet_ml` package.
```bash
pip install comet_ml
```
In this environment, create an environment variable called `COMET_API_KEY` whose value is your API key. This can be done via
```bash
export COMET_API_KEY = "your_API_key_here"
```
If the user does not want this environment variable to expire at the end of your shell session (such that it will have to be redefined in future shell sessions), then instead append the command to your `bash.rc` file (below) and restart your shell session.
```bash
echo 'export COMET_API_KEY = "your_API_key_here"' >> ~/.bashrc
```
Either way, if done correctly, the command `echo $COMET_API_KEY` will return your API key.

# Running

Edit the tracking section in `~/ClimatExML/ClimatExML/conf/config.yaml` with your project name, experiment name, and other relevant details. The next time you train, you will be able to track your experiment by going to your projects and then selecting your experiment.