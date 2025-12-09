# Capacity Tracking After Shows Close

## How It Works

The Nutcracker Trajectory Tracker automatically accounts for capacity changes as shows close throughout the season. Each city has its own closing date:

- **Edmonton**: Shows close on December 7
- **Calgary**: Shows close on December 24
- **Combined view**: Uses the later closing date (December 24)

## Understanding Show Status

The app tracks which cities have active shows and displays this information in two ways:

1. **Show Status Indicator** (on main page): Shows which cities are currently active vs. closed
2. **Summary Table**: Includes a "Shows Status" column showing "Active" or "Closed" for each city

## Handling Capacity When Shows Close

### How Show Status is Determined

The app compares **today's date** with each city's closing date:
- If today is **before or on** the closing date: Status = "Active"
- If today is **after** the closing date: Status = "Closed"

This means the status updates automatically based on the current date - no manual changes needed!

### Scenario: Edmonton Shows Have Closed

Once Edmonton shows close (after December 7), you have several options:

#### Option 1: Continue Uploading All Cities (Recommended)
Continue uploading your `this_year.csv` with both Edmonton and Calgary data:
- Edmonton rows will show "Closed" status
- Calgary rows will continue to show projections
- The app automatically handles the different closing dates

#### Option 2: Upload Only Active Cities
If Edmonton shows have completely closed and you want to simplify your data:
- Upload only Calgary sales data in `this_year.csv`
- Ensure the `Num_Shows` and `Capacity` columns reflect Calgary's totals (14 shows, 30,800 capacity)
- The app will only show projections for Calgary

## Data File Format

Your `this_year.csv` should include these columns:

```csv
Date,City,Source,Tickets Sold,Revenue,Season,Num_Shows,Capacity
2025-06-03,Calgary,Ticketmaster,532,47589,2025,14,30800
2025-06-03,Edmonton,Ticketmaster,274,25271,2025,7,15400
```

### Key Fields:

- **Num_Shows**: Total number of shows for that city's run (e.g., 14 for Calgary, 7 for Edmonton)
- **Capacity**: Total capacity for that city's run (e.g., 30,800 for Calgary, 15,400 for Edmonton)

These values should remain constant throughout the season and represent the **total** for each city, not the remaining shows.

## Per-City Tracking

The app tracks capacity independently for each city:

1. **Before Any Shows Close**
   - Both cities show projections
   - Each uses its own capacity and num_shows

2. **After Edmonton Closes (Dec 7)**
   - Edmonton shows "Closed" status
   - Edmonton projections stop at closing date
   - Calgary continues with full projections

3. **After All Shows Close (Dec 24)**
   - Both cities show "Closed" status
   - All projections complete

## Frequently Asked Questions

### Q: Do I need to update Num_Shows or Capacity after shows close?
**A: No.** Keep the values as the **total** shows and capacity for each city's run. The app uses the closing dates to determine which shows are still active.

### Q: What if I only have data for one city?
**A: That's fine.** Upload just the city with sales data. The app will project only for cities present in your data.

### Q: How does the app know when shows close?
**A: The app calculates closing dates using two pieces of information:**
1. **City-specific closing day** (hardcoded in `_city_closing_day()` function):
   - Edmonton: December 7
   - Calgary: December 24
2. **Season year** from your `Season` column (e.g., "2025" → December 7, 2025)

The app then compares today's date with each city's closing date to determine if shows are "Active" or "Closed".

### Q: Can I change the closing dates?
**A: Yes, but it requires code changes.** Edit the `_city_closing_day()` function in `app.py` to modify closing dates for specific cities.

### Q: What happens to historical data?
**A: Historical data remains unchanged.** The show status tracking only applies to the current season being projected.

## Example Workflow

### Week 1-10 (Before Edmonton closes)
```
Upload: this_year.csv with both Calgary and Edmonton data
Result: Both cities show "Active" status with projections
```

### Week 11+ (After Edmonton closes, Dec 7)
```
Option A: Upload this_year.csv with both cities
Result: 
- Edmonton shows "Closed" status
- Calgary shows "Active" status with continued projections

Option B: Upload this_year.csv with only Calgary
Result:
- Only Calgary appears with "Active" status and projections
```

### After All Shows Close (Dec 24+)
```
Upload: Final this_year.csv with all sales data
Result: Both cities show "Closed" status, projections complete
```

## Summary

The app automatically handles capacity tracking after shows close by:
1. Using city-specific closing dates
2. Displaying show status for each city
3. Calculating projections independently per city
4. Allowing flexible data uploads (all cities or only active ones)

You don't need to manually adjust capacity or num_shows values - just keep uploading your sales data and the app will handle the rest.
