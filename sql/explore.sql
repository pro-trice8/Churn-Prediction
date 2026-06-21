SELECT Contract,
       COUNT(*) AS customers,
       ROUND(AVG(CASE WHEN Churn='Yes' THEN 1.0 ELSE 0 END), 3) AS churn_rate
FROM telco
GROUP BY Contract
ORDER BY churn_rate DESC;