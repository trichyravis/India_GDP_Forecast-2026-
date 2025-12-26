# India GDP Projection 2026 - Streamlit App

**Interactive Financial Modeling Dashboard**  
The Mountain Path - World of Finance  
Prof. V. Ravichandran

---

## 🚀 Deploy in 5 Minutes

### The Easiest Way:

1. **Download 2 files:**
   - `streamlit_app.py`
   - `requirements.txt`

2. **Create GitHub repository:**
   - Go to https://github.com/new
   - Name: `india-gdp-forecast`
   - Upload both files
   - Click "Add file" → "Upload files"

3. **Deploy to Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Select file: `streamlit_app.py`
   - Click "Deploy"
   - **Done! ✅** Your app is live in 2-3 minutes

### Your app URL will be:
```
https://share.streamlit.io/YOUR_USERNAME/india-gdp-forecast/streamlit_app.py
```

---

## 📦 What's Included

### `streamlit_app.py` (Main App - 600+ lines)
**5 Interactive Pages:**

1. **Dashboard** 📊
   - Base case forecast: 6.56%
   - 3 scenarios (bull/base/bear)
   - Monte Carlo simulation (10,000 runs)
   - Statistical distribution
   - Percentile analysis

2. **Scenario Builder** ⚙️
   - 9 interactive sliders
   - Real-time GDP calculation
   - Sector breakdown chart
   - Custom forecast generation

3. **Sensitivity Analysis** 📈
   - 5 key variables
   - Tornado chart
   - Interactive shock calculator
   - Elasticity coefficients

4. **Institutional Comparison** 📊
   - RBI, IMF, ADB, Crisil, World Bank
   - Consensus analysis
   - Model positioning
   - Range comparison

5. **Download Results** 📥
   - Export Excel
   - Export CSV
   - Data preview
   - Timestamp tracking

### `requirements.txt` (Dependencies)
Lists all Python packages needed:
- streamlit 1.40.0
- pandas 2.1.4
- numpy 1.24.3
- matplotlib 3.8.2
- seaborn 0.13.0
- scikit-learn 1.3.2
- scipy 1.11.4
- openpyxl 3.11.0

---

## 💻 Run Locally (Optional)

If you want to test before deploying:

```bash
# Install packages
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py

# Opens at http://localhost:8501
```

---

## 🎯 Key Features

### Interactive Dashboard
- Real-time calculations
- Professional visualizations
- No coding needed
- Easy to use sliders
- Instant results

### Scenario Analysis
- Create custom scenarios
- Test different assumptions
- Compare outcomes
- Export for analysis

### Sensitivity Testing
- Understand key drivers
- Test variable impacts
- Make informed decisions
- Visual tornado charts

### Professional Output
- Charts and graphs
- Data tables
- Download capability
- Shareable results

---

## 📊 Model Specifications

### Base Case Forecast
- **GDP Growth: 6.56%**
- Upside (20% prob): 7.15%
- Downside (20% prob): 5.89%
- Probability-weighted: 6.65%

### Sector Breakdown
- Agriculture (18%): 3.75%
- Manufacturing (27%): 7.25%
- Services (55%): 8.75%

### Key Sensitivities
1. US Tariffs: -0.50 elasticity
2. Oil Prices: -0.10 elasticity
3. Monsoon: +0.20 elasticity
4. Global Growth: +0.30-0.50
5. Capex Growth: +0.15

### Base Case Assumptions
- Oil: $62.50/bbl
- Monsoon: 97% of normal
- Inflation: 2.5%
- Capex Growth: 35% YoY
- Tariff Impact: -10 bps

---

## 🔧 Customization (Easy!)

### Change Oil Price Assumption

In `streamlit_app.py`, find line ~360:
```python
oil_change = st.slider(
    "Oil Price Change (USD/bbl)",
    min_value=-15.0,
    max_value=25.0,
    value=0.0,        # ← Change this default
    step=0.5,
)
```

### Change Base Case Variables

Find line ~330-340:
```python
base_variables = {
    'Oil_Price_Change': 0.0,           # ← Change defaults
    'Monsoon_Deviation': 0.0,          # ← Here
    'Capex_Growth_Change': 0.0,        # ← And here
    # ...
}
```

### Update Institutional Forecasts

Find line ~1025:
```python
institutions_data = {
    'Institution': [...],
    'Forecast (%)': [6.56, 7.30, 6.40, 6.70, 7.00, 6.50],
    #                      ^^^^  ^^^^  ^^^^  ^^^^  ^^^^
    #                  Update these when new forecasts released
}
```

---

## 📱 Share Your App

### On LinkedIn:
```
Just deployed interactive India GDP forecast app! 📊

Try it here: [YOUR_APP_URL]

Features:
✅ Interactive scenario builder
✅ Sensitivity analysis with tornado charts
✅ Monte Carlo simulation (10,000 runs)
✅ Institutional forecast comparison
✅ Download results (Excel & CSV)

Build your custom 2026 forecast. What's your view? Comment below!

#India #GDP #Economics #MacroStrategy #FinanceModeling
```

### In Email:
```
Check out the interactive India GDP forecast dashboard:
[YOUR_APP_URL]

Use the Scenario Builder to test your assumptions and generate 
custom forecasts. Download results to Excel.
```

### On Your Website:
```html
<iframe
  src="https://share.streamlit.io/YOUR_USERNAME/india-gdp-forecast/streamlit_app.py"
  height="800"
  style="width: 100%;">
</iframe>
```

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Failed to deploy" | Check requirements.txt exists and is in repo root |
| "Module not found" | All packages in requirements.txt? Reinstall locally |
| "Charts don't show" | Refresh browser or check Streamlit Cloud logs |
| "Slow performance" | Reduce Monte Carlo sims from 10,000 to 5,000 |
| "Can't find file" | File path should be `streamlit_app.py` exactly |

---

## 📈 After Deployment

### Immediate (Next 24 hours)
- [ ] Test all 5 tabs
- [ ] Try Scenario Builder
- [ ] Download sample results
- [ ] Share URL with 5 people

### This Week
- [ ] Get feedback
- [ ] Customize for your use
- [ ] Share on LinkedIn
- [ ] Add to your website

### Monthly
- [ ] Monitor usage (Streamlit Cloud)
- [ ] Gather feedback
- [ ] Plan enhancements

### Quarterly (When GDP Data Releases)
- [ ] Update actual GDP data
- [ ] Recalibrate model
- [ ] Push to GitHub
- [ ] App auto-updates

---

## 🎓 For Different Audiences

### MBA/CFA Students
- Use Scenario Builder for assignments
- Study sensitivity analysis
- Learn macro modeling
- Understand elasticities

### Portfolio Managers
- Test macro scenarios
- Inform sector allocation
- Model risk impacts
- Make data-driven decisions

### Corporate Finance
- Plan capex with GDP assumptions
- Model sensitivity
- Justify budgets
- Present to board

### Investment Committees
- Discuss macro assumptions
- Validate forecasts
- Test scenarios
- Make investment decisions

---

## 💾 Updating the App

### When You Make Changes:

1. Edit `streamlit_app.py` locally
2. Test: `streamlit run streamlit_app.py`
3. Push to GitHub:
   ```bash
   git add streamlit_app.py
   git commit -m "Updated [feature]"
   git push
   ```
4. Streamlit Cloud auto-redeploys (~30 sec)
5. Your live app updates instantly!

### No downtime, no manual deployment needed!

---

## 📞 Support Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **This Guide**: Streamlit_Deployment_Guide.md
- **GitHub Issues**: Report bugs
- **Streamlit Forum**: https://discuss.streamlit.io

---

## ✨ App Highlights

✅ **Professional Design** - Mountain Path branding  
✅ **Interactive UI** - Sliders, tabs, charts  
✅ **Real-Time Calculation** - Instant forecast updates  
✅ **Scientific Rigor** - Elasticity-based modeling  
✅ **Easy Sharing** - One URL for everyone  
✅ **Free Hosting** - Streamlit Cloud  
✅ **Easy Updates** - Push to GitHub, auto-deploy  
✅ **Downloadable Results** - Excel & CSV export  

---

## 🚀 You're Ready!

**Choose Your Path:**

### Fast Lane (10 min to deployment)
1. Create GitHub repo
2. Upload 2 files
3. Deploy to Streamlit Cloud
4. Share URL

### Careful Lane (Test first)
1. Install locally
2. Run: `streamlit run streamlit_app.py`
3. Test all features
4. Then deploy

### Custom Lane (Personalize first)
1. Download files
2. Edit `streamlit_app.py`
3. Add your logo/colors/text
4. Test locally
5. Push to GitHub
6. Deploy

---

## 🎉 What Happens Next

**30 seconds after deployment:**
- Your app is live on internet
- Anyone with link can access
- No installation needed
- Professional looking dashboard

**Your URL:** `https://share.streamlit.io/YOUR_USERNAME/india-gdp-forecast/streamlit_app.py`

---

## 📊 Example Use Cases

### Internal Analysis
- Use Scenario Builder to test assumptions
- Export results for presentations
- Share with team via app link

### Client/Investor Presentations
- Demo interactive dashboard
- Show scenario analysis
- Answer "what if" questions
- Build confidence with rigor

### Thought Leadership
- Share on LinkedIn
- Embed on website
- Show expertise
- Generate engagement

### Educational
- MBA case studies
- CFA exam preparation
- Learning platform
- Interactive tool

---

## 💡 Pro Tips

1. **Test all features locally first** before going live
2. **Share the exact URL** - easier than explaining
3. **Update quarterly** when GDP data releases
4. **Monitor feedback** - users love interactive tools
5. **Add your branding** - makes it personal
6. **Document assumptions** - transparency builds trust

---

## ❓ FAQs

**Q: Is it really free?**  
A: Yes! Streamlit Cloud is free tier is unlimited. Paid options available.

**Q: How do I update forecasts?**  
A: Edit `streamlit_app.py`, push to GitHub, auto-deploys in 30 seconds.

**Q: Can I customize the colors?**  
A: Yes! Change hex codes in lines 30-50 of streamlit_app.py

**Q: How many users can access it?**  
A: Unlimited! Stream lit Cloud handles traffic automatically.

**Q: Can I embed in my website?**  
A: Yes, use iframe embedding in HTML.

**Q: Will it handle millions of users?**  
A: Free tier: up to 3GB RAM. Paid tier if you need more.

---

**Ready to deploy? Go to https://share.streamlit.io and follow the Deployment Guide! 🚀**

---

*The Mountain Path - World of Finance*  
*Prof. V. Ravichandran*  
*28+ Years Corporate Finance Experience*  
*10+ Years Academic Excellence*

**Questions? Read Streamlit_Deployment_Guide.md for detailed instructions!**
