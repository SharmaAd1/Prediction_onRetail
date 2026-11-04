# Retail Revenue Forecast

Streamlit dashboard for exploring retail revenue and forecasting daily revenue
with Prophet, with Gemini-powered answers about the current dashboard filters
and forecast.

## Run locally

From the repository root:

```bash
python3 -m pip install -r newappProject/requirements.txt
python3 -m streamlit run newappProject/app.py
```

Without MySQL configured, the app reads `newappProject/retail_data_c.csv` from
beside its entrypoint.

## Connect to MySQL

The app uses MySQL automatically when MySQL environment variables or a
`[mysql]` section in Streamlit secrets are configured; otherwise it uses the
local CSV. The table must contain the same columns as `retail_data_c.csv`,
including `InvoiceDate`, `Revenue`, and the other sales fields.

For local development, create `newappProject/.env` (it is ignored by Git):

```dotenv
MYSQL_HOST=your-mysql-host
MYSQL_PORT=3306
MYSQL_DATABASE=your-database
MYSQL_USER=your-user
MYSQL_PASSWORD=your-password
MYSQL_TABLE_NAME=retail_sales
```

The same values can be supplied as environment variables. For Community Cloud,
use **Advanced settings > Secrets**:

```toml
[mysql]
host = "your-mysql-host"
port = 3306
database = "your-database"
user = "your-user"
password = "your-password"
table_name = "retail_sales"
```

Environment variables take precedence over Streamlit secrets. Never commit
real credentials. MySQL credentials are only used to read the sales table.

## Ask Gemini about the forecast

The dashboard uses Google's
[Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models) model to
answer questions about the selected category, region, date range, and forecast.
Configure `GEMINI_API_KEY` as a top-level Streamlit secret or environment
variable:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

For local runs, save the secret in `newappProject/.streamlit/secrets.toml` and
launch Streamlit from `newappProject` so it can discover the file.

Only aggregate historical metrics, up to 24 monthly revenue totals, and the
daily forecast estimates are sent to Gemini. Customer-level records and
database credentials are not included. The Gemini API is billed separately
according to your Google AI plan.

## Deploy to Streamlit Community Cloud

1. Push the app files to a GitHub branch:
   - `newappProject/app.py`
   - `newappProject/db_connect.py`
   - `newappProject/requirements.txt`
   - `newappProject/.gitignore`
   - This README (optional)
   - `newappProject/retail_data_c.csv` if you are using the CSV instead of MySQL
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**,
   then select the GitHub repository and branch.
3. Set **Main file path** to `newappProject/app.py`.
4. In **Advanced settings**, choose Python 3.12, then deploy.

The repository already has a root `.streamlit/config.toml`; Community Cloud
uses that configuration for this app. If MySQL is not configured, include the
CSV dataset in the GitHub repository.

The app and included dataset may be accessible to anyone who can access the
repository or deployed app. Do not deploy private or sensitive data publicly.
