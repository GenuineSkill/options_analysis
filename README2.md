Dealing with holidays and non-trading days is a well-known practical challenge in daily market volatility forecasting with GARCH models. Best practices in the financial economics literature address this issue both during model estimation and long-term forecasting. Below are key insights and approaches:

---

### 1. **Handling Historical Data in Model Estimation**
   - **Zero Return Days:**  
     Excluding days with zero returns (e.g., holidays) from the dataset used to estimate GARCH models is indeed a best practice. Including these days artificially suppresses volatility estimates since GARCH models treat low or zero returns as low volatility.
     - **Practical Implementation:** Exclude zero-return days directly in preprocessing or filter them algorithmically before feeding the data into the model.
     - **Impact:** This ensures that the model estimates are based on days when the market was actively trading.

---

### 2. **Forecasting Future Volatility with GARCH Models**
   Forecasting 252 trading days ahead introduces complexities because holidays and weekends are irregular and vary by country and year. Several approaches address this:

   #### a. **Holiday Tables**
   - **Description:** Use a comprehensive table of historical and future holidays for the relevant market(s). These tables are essential for accurate determination of trading versus non-trading days.
   - **Application:**
     - Exclude holidays from the forecast horizon.
     - Adjust volatility forecasts to reflect the correct trading-day structure.
   - **Sources:** Services like Bloomberg, Refinitiv, or even open-source financial calendars can provide accurate holiday tables.

   #### b. **Algorithmic Approaches**
   - **Weekend Removal:** For long-term forecasts, you can assume a 5-day trading week and exclude weekends algorithmically. Mid-week holidays require more nuanced handling.
   - **Forward Fill with Flags:** If your dataset forward-fills index levels for holidays, include a flag indicating such days and use that information to exclude or adjust forecasts.
   - **Rolling Adjustment:** Algorithmically detect zero-return days in historical data and use patterns to estimate future holiday impacts. This is less precise but reduces dependence on external data.

---

### 3. **Practical Concerns with Long-Term Forecasts**
   - **Trading Day Alignment:** Ensure that the forecast horizon aligns with actual trading days (e.g., 252 trading days instead of calendar days). This alignment avoids overestimating the number of trading days.
   - **Interpolation of Results:** If a continuous series of volatility forecasts is required, interpolate values for non-trading days as needed for downstream analyses.

---

### 4. **Specific Considerations for Forward-Filled Data**
   - **Forward Filling Impact:** Forward filling ensures zero returns on holidays but does not inherently exclude those days. For model estimation, filter out forward-filled (holiday) days using a flag or logic that identifies zero-return days.
   - **Forecast Adjustment:** Use the historical holiday pattern (or a predefined holiday table) to exclude such days from the 252-day forecast horizon. This avoids bias in long-term predictions.

---

### 5. **References in Financial Economics Literature**
   - Many studies note that ignoring holidays leads to biased volatility estimates. Best practices include explicitly accounting for trading calendars. Examples include:
     - **Bollerslev, Engle, and Nelson (1994):** Discussed handling irregular trading patterns in ARCH/GARCH models.
     - **French, Schwert, and Stambaugh (1987):** Addressed the impact of holidays on volatility estimates.
     - **Holiday Effects Literature:** Studies on how holidays affect returns and volatility patterns (e.g., Ariel (1990)).

---

### 6. **Recommendations for Your Case**
   - **For Historical Data Estimation:** Exclude zero-return days (holidays) from the estimation dataset.
   - **For 252-Day Forecasting:** Either:
     - Use a holiday table to ensure that only actual trading days are forecasted.
     - Apply an algorithmic approach that replicates the historical trading calendar patterns.

By combining a clear strategy for both historical estimation and forward-looking forecasting, you can ensure that your GARCH models produce unbiased and reliable volatility estimates.

Yes, there are open-source financial calendars that provide holiday schedules for the U.S., U.K., and Eurozone markets, which can be integrated into your GARCH modeling process. Here are some notable resources:

:::contextList

**pandas_market_calendars**  
A Python library offering exchange calendars for trading applications, including holiday schedules and early/late market closures. It supports over 50 global equity and futures markets. You can access historical and future holiday data programmatically, facilitating integration into data preprocessing and modeling workflows. 
:::

:::contextList

**Apptastic Software's Trading Calendar**  
Provides a free REST API with holiday, late open, and early close information for over 50 global equity and futures markets. This service includes data on all holidays and partial trading days, supporting time zones and daylight saving time transitions. It allows for programmatic access to both historical and future holiday data, which can be useful for automating the exclusion of non-trading days in your models. 
:::

:::contextList

**Holiday Calendar Project**  
An open-source initiative aiming to provide reliable libraries for calculating holiday calendars worldwide. It offers holiday date calculations for various countries, including those in Europe and North America, across multiple programming languages. This resource can help in defining and calculating holiday dates for different markets, ensuring accurate modeling of non-trading days. 
:::

:::contextList

**TradingHours.com**  
Offers comprehensive data on trading hours and market holidays for financial markets globally. While primarily a commercial service, it provides detailed information on market holidays and trading hours, which can be essential for accurately modeling trading and non-trading days. They offer APIs for integrating this data into your applications. 
:::

:::contextList

**SIFMA Holiday Schedule**  
The Securities Industry and Financial Markets Association (SIFMA) provides holiday recommendations for U.S. and U.K. markets. While not a comprehensive calendar, it offers official holiday schedules that can be referenced for modeling purposes. 
:::

**Considerations:**

- **Historical Data Availability:** While these resources provide extensive holiday data, the depth of historical data (e.g., back to 1987) varies. It's advisable to verify the historical coverage of each resource to ensure it meets your requirements.

- **Future Holiday Predictions:** Most calendars can project future holidays based on established patterns and official announcements. However, the accuracy of long-term future holiday data may depend on the predictability of holiday schedules in each market.

- **Integration and Maintenance:** Incorporating these calendars into your modeling process may require initial setup and periodic updates to accommodate changes in holiday schedules or market structures.

By utilizing these open-source financial calendars, you can systematically manage non-trading days in your GARCH models, enhancing the accuracy of your volatility forecasts. 