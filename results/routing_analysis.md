# Routing analysis

Thresholds chosen on the validation set (target: 97% intent accuracy on auto-routed tickets):

- Auto-route at confidence >= **0.90**
- Human triage below **0.50**

## Test-set results with these thresholds

Overall intent accuracy: 0.9263 | overall team accuracy: 0.9718 | mean confidence: 0.9608

| Decision | Share of tickets | Intent accuracy | Team accuracy |
|---|---|---|---|
| auto-route | 90.3% | 0.968 | 0.988 |
| route + needs review | 8.0% | 0.576 | 0.865 |
| human triage | 1.8% | 0.389 | 0.593 |

## Estimated monthly impact (ASSUMPTIONS, not measurements)

Assumptions: 10,000 tickets/month; 2.0 min manual triage; 0.5 min to check a flagged ticket; 10.0 min to fix a wrong-team ticket; INR 300/hour.

- Manual triage today: 333 hours/month
- Estimated net time saved: 304 hours/month (91%)
- Estimated net saving: INR 91,063/month
- Auto-routed tickets sent to the wrong team: about 104/month
