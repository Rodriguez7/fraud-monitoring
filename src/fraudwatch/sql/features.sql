-- Feature pipeline. Input tables: transactions, customers. Output table: features.
--
-- Every window looks strictly BACKWARD (EXCLUDE CURRENT ROW, never FOLLOWING), so a
-- transaction's features only use information available when it was authorised.
-- (customer_id, ts) is unique, so RANGE frames never include same-instant "peers".

CREATE OR REPLACE TABLE features AS
WITH seen AS (
    SELECT
        t.*,
        c.home_country, c.account_age_days,
        ROW_NUMBER() OVER (PARTITION BY t.customer_id, t.merchant_id ORDER BY t.ts)    AS merchant_n,
        -- first time this customer was seen with this device / country / payee
        MIN(t.ts) OVER (PARTITION BY t.customer_id, t.device_id ORDER BY t.ts)          AS device_first,
        MIN(t.ts) OVER (PARTITION BY t.customer_id, t.merchant_country ORDER BY t.ts)   AS country_first,
        MIN(t.ts) OVER (PARTITION BY t.customer_id, t.payee_id ORDER BY t.ts)           AS payee_first
    FROM transactions t
    JOIN customers c USING (customer_id)
),
base AS (
    SELECT
        s.txn_id, s.ts, s.customer_id, s.txn_type, s.amount_eur, s.merchant_id,
        s.merchant_category, s.merchant_country, s.channel, s.device_id, s.payee_id,
        s.is_fraud, s.fraud_type, s.fraud_event_id,
        s.home_country, s.account_age_days,
        (s.txn_type = 'bank_transfer')                    AS is_transfer,
        (s.channel = 'online')                            AS is_online,
        (s.channel = 'card_present')                      AS is_card_present,
        (s.txn_type = 'card_payment'
            AND s.merchant_country <> s.home_country)     AS is_cross_border,
        EXTRACT(hour FROM s.ts)                           AS hour,
        EXTRACT(isodow FROM s.ts)                         AS dow,
        (EXTRACT(hour FROM s.ts) < 5)                     AS is_night,
        (s.txn_type = 'bank_transfer'
            AND s.amount_eur BETWEEN 9000 AND 9999.99)    AS is_near_threshold,
        CASE s.merchant_category
            WHEN 'gift_cards' THEN 3 WHEN 'crypto_exchange' THEN 3 WHEN 'electronics' THEN 2
            WHEN 'transfer' THEN 1 WHEN 'online_services' THEN 1 ELSE 0
        END                                               AS category_risk,
        -- "new" = first seen with this customer within the last 24 hours
        (s.merchant_id IS NOT NULL AND s.merchant_n = 1)  AS is_new_merchant,
        (s.device_id IS NOT NULL
            AND epoch(s.ts - s.device_first) < 86400)     AS is_new_device,
        (epoch(s.ts - s.country_first) < 86400)           AS is_new_country,
        (s.payee_id IS NOT NULL
            AND epoch(s.ts - s.payee_first) < 86400)      AS is_new_payee
    FROM seen s
),
cp AS (
    -- previous card-present payment per customer, for the impossible-travel check
    SELECT
        txn_id,
        LAG(merchant_country) OVER w AS prev_cp_country,
        LAG(ts) OVER w               AS prev_cp_ts
    FROM base
    WHERE is_card_present
    WINDOW w AS (PARTITION BY customer_id ORDER BY ts)
),
win AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER w_all - 1                                    AS customer_txn_seq,
        epoch(b.ts - LAG(b.ts) OVER w_all)                             AS secs_since_prev,
        COUNT(*) OVER w10m                                             AS cnt_10m,
        COUNT(*) OVER w1h                                              AS cnt_1h,
        COUNT(*) OVER w24h                                             AS cnt_24h,
        COALESCE(SUM(b.amount_eur) OVER w24h, 0)                       AS sum_24h,
        AVG(b.amount_eur) OVER w30d                                    AS avg_30d,
        COALESCE(STDDEV_SAMP(b.amount_eur) OVER w30d, 0)               AS std_30d,
        MAX(b.amount_eur) OVER w30d                                    AS max_30d,
        COALESCE(SUM(CASE WHEN b.is_transfer THEN 1 ELSE 0 END) OVER w24h, 0)
                                                                       AS transfers_24h,
        COALESCE(SUM(CASE WHEN b.is_transfer AND b.is_new_payee THEN 1 ELSE 0 END) OVER w24h, 0)
                                                                       AS new_payee_transfers_24h,
        COALESCE(SUM(CASE WHEN b.is_near_threshold THEN 1 ELSE 0 END) OVER w48h, 0)
                                                                       AS near_threshold_48h
    FROM base b
    WINDOW
        w_all AS (PARTITION BY b.customer_id ORDER BY b.ts),
        w10m  AS (PARTITION BY b.customer_id ORDER BY b.ts
                  RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
        w1h   AS (PARTITION BY b.customer_id ORDER BY b.ts
                  RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
        w24h  AS (PARTITION BY b.customer_id ORDER BY b.ts
                  RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
        w48h  AS (PARTITION BY b.customer_id ORDER BY b.ts
                  RANGE BETWEEN INTERVAL 48 HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
        w30d  AS (PARTITION BY b.customer_id ORDER BY b.ts
                  RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW)
)
SELECT
    w.txn_id, w.ts, w.customer_id, w.txn_type, w.merchant_category, w.merchant_country,
    w.channel, w.device_id, w.payee_id, w.home_country,
    w.is_fraud, w.fraud_type, w.fraud_event_id,
    w.amount_eur,
    w.is_transfer::INTEGER AS is_transfer, w.is_online::INTEGER AS is_online,
    w.is_card_present::INTEGER AS is_card_present, w.is_cross_border::INTEGER AS is_cross_border,
    w.hour, w.dow, w.is_night::INTEGER AS is_night, w.category_risk,
    w.is_near_threshold::INTEGER AS is_near_threshold,
    w.is_new_merchant::INTEGER AS is_new_merchant, w.is_new_device::INTEGER AS is_new_device,
    w.is_new_country::INTEGER AS is_new_country, w.is_new_payee::INTEGER AS is_new_payee,
    w.account_age_days, w.customer_txn_seq,
    COALESCE(w.secs_since_prev, 1e9) AS secs_since_prev,
    w.cnt_10m, w.cnt_1h, w.cnt_24h, w.sum_24h,
    COALESCE(w.avg_30d, w.amount_eur) AS avg_30d,
    w.std_30d,
    COALESCE(w.max_30d, w.amount_eur) AS max_30d,
    (w.amount_eur - COALESCE(w.avg_30d, w.amount_eur)) / (w.std_30d + 10.0) AS amount_z,
    w.amount_eur / COALESCE(w.max_30d, w.amount_eur) AS amount_ratio_max,
    w.transfers_24h, w.new_payee_transfers_24h, w.near_threshold_48h,
    (cp.prev_cp_country IS NOT NULL
        AND cp.prev_cp_country <> w.merchant_country
        AND epoch(w.ts - cp.prev_cp_ts) < 7200)::INTEGER AS country_changed_fast
FROM win w
LEFT JOIN cp USING (txn_id)
ORDER BY w.ts;
