# 🔍 CompIq - Agentic AI Comparables Finder

<img src="compiq.png" alt="CompIQ Logo" width="550"/>

An intelligent, agentic system for finding publicly-traded comparable companies using AI-powered analysis.

![Demo](https://img.shields.io/badge/Status-Demo%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## 🎯 Features

- **AI-Powered Analysis**: Uses GPT-4 to deeply understand your target company
- **Dynamic Validation**: Automatically verifies companies are currently publicly traded
- **Semantic Matching**: Embeddings-based similarity scoring for accurate comparables
- **Real-time Progress**: Watch the agent work with live progress updates
- **Persistent Storage**: SQLite database tracks all searches and results
- **Export Options**: Download results as CSV or JSON
- **Search History**: Quickly revisit previous analyses

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this project
cd comparables-finder

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set your OpenAI API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY='your-api-key-here'

# Option 2: Enter in the Streamlit sidebar
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Finding Comparables

1. **Enter Target Company Details**
   - Company name
   - Detailed business description
   - Homepage URL (optional)
   - Primary SIC code (optional)

2. **Configure Settings** (sidebar)
   - Minimum comparables required (default: 3)
   - Maximum comparables to return (default: 10)
   - Max search attempts (default: 3)

3. **Run Search**
   - Click "Find Comparables"
   - Watch the agent work in real-time
   - Review results and export

### Understanding Results

Each comparable is scored on:
- **Semantic Similarity**: How closely the business descriptions match
- **Focus Area Overlap**: Alignment on key industry terms
- **Business Model Match**: Similarity in revenue model and operations

**Score Interpretation:**
- 5.0+: Excellent match
- 3.0-4.9: Good match
- 2.0-2.9: Fair match
- <2.0: Weak match

### Using the Database

The app automatically saves:
- All search queries and results
- Validated company information
- Search history for quick recall

Access previous searches in the sidebar to reload results instantly.

## 🏗️ Architecture

```
comparables-finder/
├── app.py              # Streamlit UI
├── comps_agent.py      # Core AI agent logic
├── database.py         # SQLite persistence
├── requirements.txt    # Dependencies
└── comparables.db      # Auto-generated database
```

### Key Components

**ComparablesAgent** (`comps_agent.py`)
- Target company analysis
- Candidate generation
- Semantic similarity scoring
- Public status validation

**Database** (`database.py`)
- Search history management
- Company information caching
- Results persistence

**Streamlit App** (`app.py`)
- Interactive UI
- Real-time progress tracking
- Results visualization
- Export functionality

## 🔧 Technical Details

### AI Models Used
- **GPT-4**: Analysis, candidate generation, validation
- **text-embedding-3-small**: Semantic similarity matching

### Validation Process
1. **Data Validation**: Check for complete company information
2. **Public Status Check**: Verify currently trading (not acquired/delisted)
3. **Operating Company Check**: Exclude holding companies/SPACs
4. **Business Model Match**: Assess revenue model alignment
5. **Semantic Scoring**: Calculate similarity via embeddings

### Dynamic, Industry-Agnostic Design
The system uses **zero hardcoded industry logic**. All validation and matching is performed dynamically via LLM reasoning, making it work across:
- Technology (software, hardware, semiconductors)
- Healthcare (biotech, medical devices, services)
- Financial services (fintech, insurance, banking)
- Industrial (manufacturing, construction, materials)
- Emerging sectors (quantum computing, space tech, carbon capture)

## 📊 Example Queries

### Tech Company
```
Name: Scale AI
Description: Provides data labeling and curation services for AI training...
```

### Healthcare
```
Name: PathAI Diagnostics
Description: AI-powered digital pathology for cancer diagnosis...
```

### Fintech
```
Name: ClearLedger Technologies
Description: Digital payments infrastructure for banks...
```

## 🎨 Customization

### Adjust Scoring Weights
Edit `_score_comparable()` in `comps_agent.py`:
```python
# Increase semantic similarity weight
weight = 5.0 + (specialization * 2.0)  # Default: 3.0

# Adjust focus area importance
score += focus_score * 2.0  # Default: 1.5
```

### Modify Search Strategy
Edit `_generate_candidates()` in `comps_agent.py` to change search prompts and strategies.

### Change UI Theme
Edit CSS in `app.py` or use Streamlit's built-in themes:
```bash
streamlit run app.py --theme.base="dark"
```

## 🐛 Troubleshooting

### "No candidates generated"
- **Cause**: Target description may be too vague
- **Fix**: Add more specific details about products, customers, and business model

### "All candidates rejected"
- **Cause**: Very niche industry with few public comparables
- **Fix**: System will automatically broaden search; consider adjusting min_required

### Slow performance
- **Cause**: OpenAI API rate limits or network latency
- **Fix**: Reduce batch sizes in `_verify_batch()` or add longer delays

## 📈 Future Enhancements

Potential additions:
- [ ] Excel export with formatted reports
- [ ] Financial data integration (revenue, multiples)
- [ ] Company comparison matrix
- [ ] Web scraping for real-time verification
- [ ] Advanced filtering (market cap, geography, etc.)
- [ ] Collaborative features (team sharing)
- [ ] API endpoint for programmatic access

## 🤝 Contributing

This is a personal project, but suggestions welcome! Key areas for contribution:
- Additional validation heuristics
- UI/UX improvements
- Export format options
- Performance optimizations

## 📝 License

MIT License - feel free to use and modify for your own projects.

## 🙏 Acknowledgments

Built using:
- [Streamlit](https://streamlit.io/) - Web framework
- [OpenAI](https://openai.com/) - AI models
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [SQLite](https://www.sqlite.org/) - Database

---
