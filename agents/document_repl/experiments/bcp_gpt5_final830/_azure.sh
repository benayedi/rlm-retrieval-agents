# Sourced by the launchers in this directory. Credential wiring only.
# The original run_experiment launches took the Azure credentials from <working directory>/.env, which defined
# AZURE2_ENDPOINT / AZURE2_API_KEY / AZURE2_API_VERSION (read by run_one.py and judge.py). Here they come from AZ_ENV.
# The API version of the original .env is not recorded anywhere: set AZURE2_API_VERSION explicitly.
set -a; . "${AZ_ENV:?set AZ_ENV}"; set +a
export AZURE2_ENDPOINT="$AZ_EP/" AZURE2_API_KEY="$AZ_KEY" \
       AZURE2_API_VERSION="${AZURE2_API_VERSION:?set AZURE2_API_VERSION (value used by the original runs: UNKNOWN)}"
export PYTHONPATH=$RLMLIB_PATH${PYTHONPATH:+:$PYTHONPATH}   # the original workers imported rlm from their own venv
DEP=${AZURE_DEPLOYMENT_GPT5:?set AZURE_DEPLOYMENT_GPT5}     # original: deployment "gpt-5"
