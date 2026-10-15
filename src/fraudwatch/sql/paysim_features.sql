-- Feature pipeline for the PaySim benchmark. Input table: paysim_raw (the Kaggle CSV).
-- Output table: paysim_features.
--
-- PaySim has no customer history: almost every origin account appears once. Look-back
-- features are therefore built on the DESTINATION account (mules receive many transfers),
-- using strictly earlier steps (1 PRECEDING), so same-step rows never see each other.
--
-- The "balance" columns exploit simulator accounting artefacts (for example fraud that
-- empties the origin account). They are kept separate so the benchmark can show how much
-- of the performance comes from artefacts versus behaviour.

CREATE OR REPLACE TABLE paysim_features AS
WITH src AS (
    SELECT
        row_number() OVER ()      AS rid,
        step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        nameDest, oldbalanceDest, newbalanceDest,
        CAST(isFraud AS INTEGER)        AS is_fraud,
        CAST(isFlaggedFraud AS INTEGER) AS is_flagged
    FROM paysim_raw
)
SELECT
    rid, step, type, is_fraud, is_flagged, amount,
    -- behavioural features
    CAST(type = 'TRANSFER' AS INTEGER) AS is_transfer,
    CAST(type = 'CASH_OUT' AS INTEGER) AS is_cash_out,
    CAST(type = 'PAYMENT'  AS INTEGER) AS is_payment,
    CAST(type = 'CASH_IN'  AS INTEGER) AS is_cash_in,
    CAST(type = 'DEBIT'    AS INTEGER) AS is_debit,
    LN(amount + 1)                      AS log_amount,
    (step - 1) % 24                     AS hour_of_day,
    CAST(nameDest LIKE 'M%' AS INTEGER) AS dest_is_merchant,
    COUNT(*) OVER w24                   AS dest_cnt_24h,
    COALESCE(SUM(amount) OVER w24, 0)   AS dest_sum_24h,
    COUNT(*) OVER w168                  AS dest_cnt_7d,
    -- balance / accounting features (simulator artefacts)
    oldbalanceOrg                       AS orig_balance_before,
    CAST(newbalanceOrig = 0 AND oldbalanceOrg > 0 AS INTEGER) AS orig_emptied,
    amount / (oldbalanceOrg + 1)        AS amount_to_balance,
    oldbalanceOrg - amount - newbalanceOrig AS orig_balance_error,
    oldbalanceDest                      AS dest_balance_before,
    oldbalanceDest + amount - newbalanceDest AS dest_balance_error
FROM src
WINDOW
    w24  AS (PARTITION BY nameDest ORDER BY step RANGE BETWEEN 24  PRECEDING AND 1 PRECEDING),
    w168 AS (PARTITION BY nameDest ORDER BY step RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING);
