# Betting Scraper Implementation Notes

## What Was Created

A complete betting odds scraping system with:

1. **Data Types** (`src/utils/data_types.py`):
   - `BettingOption` - Represents individual betting options from websites
   - `to_proposition()` method - Converts betting options to analyzable propositions

2. **Scrapers**:
   - `SleeperScraper` (`src/data/collectors/sleeper_scraper.py`)
   - `UnderdogScraper` (`src/data/collectors/underdog_scraper.py`)
   - `ESPNScraper` (`src/data/collectors/espn_scraper.py`)

3. **Unified Manager** (`src/data/collectors/betting_scraper_manager.py`):
   - `BettingScraperManager` - Single interface for all scrapers
   - Methods to get bets, search, and format for display

4. **CLI Commands** (`src/cli.py`):
   - `browse` - List betting options from a site
   - `analyze-bet` - Analyze a specific bet option

## Important: Template Implementation

⚠️ **The scrapers are template implementations** that need to be customized based on the actual website structures.

### Why Templates?

Each betting website has:
- Different HTML structures
- Different API endpoints (if any)
- Different authentication requirements
- Different data formats

The current implementation provides:
- ✅ Complete framework and structure
- ✅ Data flow and integration
- ✅ CLI interface
- ⚠️ Placeholder parsing logic that needs real website inspection

### What Needs Customization

For each scraper, you'll need to:

1. **Inspect the actual website**:
   - Open browser developer tools
   - Navigate to a game's betting page
   - Identify HTML structure
   - Find API endpoints (if available)

2. **Update parsing methods**:
   - `_parse_bet_from_html()` - Extract bet data from HTML
   - `_find_game_id()` - Locate games on the site
   - `_get_bets_for_game()` - Fetch bets for a specific game

3. **Test with real data**:
   - Verify bets are extracted correctly
   - Check odds format matches
   - Ensure all bet types are captured

## How to Customize a Scraper

### Step 1: Inspect the Website

1. Open the betting site in a browser
2. Navigate to an NFL game's betting page
3. Open Developer Tools (F12)
4. Inspect the HTML structure of betting options

### Step 2: Identify Key Elements

Look for:
- Container elements for betting options
- Title/description elements
- Odds display elements
- Player name elements (for props)
- Line value elements (for spreads/totals)

### Step 3: Update the Scraper

Example for Sleeper:

```python
def _parse_bet_from_html(self, element, rank: int) -> Optional[BettingOption]:
    """Parse a bet option from HTML element."""
    try:
        # Update these selectors based on actual HTML structure
        title_elem = element.find('div', class_='bet-title')  # Actual class name
        title = title_elem.text.strip() if title_elem else "Unknown Bet"
        
        odds_elem = element.find('span', class_='odds-value')  # Actual class name
        odds_str = odds_elem.text.strip() if odds_elem else "-110"
        odds = self._parse_odds(odds_str)
        
        # ... rest of parsing logic
```

### Step 4: Test

```bash
# Test the scraper
python -m src.cli browse \
    --site sleeper \
    --home-team NYG \
    --away-team NE \
    --date 2024-12-15
```

## API vs Web Scraping

### API Approach (Preferred)

If a site has an API:
1. Find the API endpoint
2. Check authentication requirements
3. Update `_get_bets_for_game()` to use API
4. Parse JSON response in `_parse_bet_option()`

### Web Scraping Approach (Fallback)

If no API available:
1. Use BeautifulSoup to parse HTML
2. Find the correct CSS selectors
3. Extract data from HTML elements
4. Handle dynamic content (may need Selenium)

## Common Patterns

### Finding Games

Most sites have:
- A games listing page
- Game cards/links with team names
- Game IDs in URLs or data attributes

### Finding Bets

Most sites display bets as:
- Cards or list items
- Grouped by bet type
- With odds displayed prominently

### Parsing Odds

Odds are typically:
- American format: `-110`, `+150`
- Sometimes with text: `-110 (1.91)`
- May need cleaning: `remove_currency_symbols()`

## Testing Checklist

Before using a scraper in production:

- [ ] Can find games by team names
- [ ] Can extract at least 10 different bets
- [ ] Odds are parsed correctly
- [ ] Bet types are classified correctly
- [ ] Player names extracted (for props)
- [ ] Line values extracted (for spreads/totals)
- [ ] Handles missing data gracefully
- [ ] Works for multiple different games
- [ ] Error handling for network issues
- [ ] Rate limiting to avoid blocking

## Example: Real Sleeper Implementation

If Sleeper's actual structure is:

```html
<div class="bet-card" data-bet-id="12345">
    <h3 class="bet-title">NY Giants -3.5</h3>
    <span class="odds">-110</span>
    <div class="bet-details">
        <span class="line">-3.5</span>
    </div>
</div>
```

Then update:

```python
def _parse_bet_from_html(self, element, rank: int) -> Optional[BettingOption]:
    title_elem = element.find('h3', class_='bet-title')
    odds_elem = element.find('span', class_='odds')
    line_elem = element.find('span', class_='line')
    
    # ... rest of implementation
```

## Maintenance

Websites change frequently. You may need to:

1. **Monitor for failures**: Set up alerts if scrapers fail
2. **Update selectors**: When sites change HTML structure
3. **Handle new bet types**: Add classification logic
4. **Update team mappings**: If team abbreviations change

## Legal and Ethical Considerations

- ✅ Check website Terms of Service
- ✅ Respect rate limits
- ✅ Don't overload servers
- ✅ Use for personal/research purposes
- ✅ Consider official APIs if available
- ⚠️ Some sites may prohibit scraping

## Support

If you need help customizing scrapers:

1. Inspect the website structure
2. Share the HTML structure (relevant parts)
3. Update the scraper code accordingly
4. Test thoroughly before production use

## Current Status

- ✅ Framework complete
- ✅ Integration with analysis pipeline complete
- ✅ CLI interface complete
- ⚠️ Scrapers need real website inspection and customization
- ⚠️ API endpoints may need authentication setup
- ⚠️ HTML selectors need to be determined from actual sites

The system is ready to use once the scrapers are customized for the actual website structures!

