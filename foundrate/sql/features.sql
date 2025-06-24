-- Point-in-time feature snapshot. Only order lines strictly BEFORE $as_of are used,
-- so a snapshot never sees the labels it will later be trained against.
WITH base AS (
    SELECT l.store_id, l.item_id, l.order_date, l.found, i.category, i.is_perishable
    FROM order_lines l
    JOIN items i ON i.item_id = l.item_id
    WHERE l.order_date < CAST($as_of AS DATE)
      AND l.order_date >= CAST($as_of AS DATE) - $window_days
),
store_item AS (
    SELECT store_id, item_id,
           ANY_VALUE(category) AS category,
           ANY_VALUE(is_perishable) AS is_perishable,
           AVG(found) AS si_rate_30d,
           COUNT(*) AS si_n_30d,
           AVG(found) FILTER (WHERE order_date >= CAST($as_of AS DATE) - 7) AS si_rate_7d,
           COUNT(*) FILTER (WHERE order_date >= CAST($as_of AS DATE) - 7) AS si_n_7d,
           COALESCE(date_diff('day', MAX(order_date) FILTER (WHERE found = 0), CAST($as_of AS DATE)), $window_days)
               AS days_since_miss
    FROM base GROUP BY store_id, item_id
),
store_7d AS (
    SELECT store_id, AVG(found) AS store_rate_7d
    FROM base WHERE order_date >= CAST($as_of AS DATE) - 7 GROUP BY store_id
),
store_cat AS (
    SELECT store_id, category, AVG(found) AS store_cat_rate_30d FROM base GROUP BY store_id, category
),
item_30d AS (
    SELECT item_id, AVG(found) AS item_rate_30d FROM base GROUP BY item_id
)
SELECT CAST($as_of AS DATE) AS as_of,
       si.store_id, si.item_id,
       si.si_rate_30d, CAST(si.si_n_30d AS DOUBLE) AS si_n_30d,
       si.si_rate_7d, CAST(COALESCE(si.si_n_7d, 0) AS DOUBLE) AS si_n_7d,
       CAST(si.days_since_miss AS DOUBLE) AS days_since_miss,
       s7.store_rate_7d, sc.store_cat_rate_30d, i30.item_rate_30d,
       CAST(si.is_perishable AS DOUBLE) AS is_perishable
FROM store_item si
LEFT JOIN store_7d s7 ON s7.store_id = si.store_id
LEFT JOIN store_cat sc ON sc.store_id = si.store_id AND sc.category = si.category
LEFT JOIN item_30d i30 ON i30.item_id = si.item_id
