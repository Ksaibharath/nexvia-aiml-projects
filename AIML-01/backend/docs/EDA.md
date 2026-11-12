# EDA — IEEE-CIS Fraud Detection

## Fraud rate
- Overall ~3.5% (20k/590k). Train/val/test each ~3.4-3.6% (compute exact in prepare.py).
- 'Always legit' accuracy = 96.5% with zero fraud caught → never use accuracy.

## Categories
- ProductCD H/S/W higher fraud than P/C. card4/card6 and rare email domains higher.
- DeviceType: unknown devices higher fraud.

## Amount & time
- Fraud amounts larger on average (log scale). Night hours (relative) higher fraud.
- Weekly fraud rate stable (no strong trend).

## Missing values & identity
- ~76% missing identity (DeviceInfo). Fraud rate differs with vs without identity → missingness is a signal.

## Accounts
- Proxy uid from card1+addr1+start-day. Typical account has few transactions; ~X% accounts have any fraud.
- LIMITATION: uid is a proxy; IEEE-CIS has no real customer id.

## Plots
See docs/eda/ (fraud_by_week.png, fraud_by_hour.png, amount_box.png, ...)
