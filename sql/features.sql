SELECT
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,

    -- Clean TotalCharges: blank strings become NULL, then we handle them
    CASE
        WHEN TRIM(CAST(TotalCharges AS VARCHAR)) = '' THEN MonthlyCharges
        ELSE CAST(TotalCharges AS DOUBLE)
    END AS TotalCharges,

    -- Engineered feature 1: spend per month of tenure
    ROUND(
        CASE
            WHEN TRIM(CAST(TotalCharges AS VARCHAR)) = '' THEN MonthlyCharges
            ELSE CAST(TotalCharges AS DOUBLE)
        END / NULLIF(tenure, 0), 2
    ) AS charges_per_tenure,

    -- Engineered feature 2: tenure bucket
    CASE
        WHEN tenure <= 12 THEN '0-1yr'
        WHEN tenure <= 24 THEN '1-2yr'
        WHEN tenure <= 48 THEN '2-4yr'
        ELSE '4-6yr'
    END AS tenure_group,

    -- Engineered feature 3: the strongest churn signal as a flag
    CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END AS is_month_to_month,

    -- Target as 0/1
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS Churn

FROM telco;