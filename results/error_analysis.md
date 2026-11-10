# DistilBERT error analysis

## Most confused intent pairs (true -> predicted)

- declined_transfer -> declined_card_payment: 5 times
- verify_my_identity -> why_verify_identity: 4 times
- top_up_reverted -> top_up_failed: 4 times
- pin_blocked -> change_pin: 4 times
- get_disposable_virtual_card -> getting_virtual_card: 4 times
- fiat_currency_support -> exchange_via_app: 4 times
- card_arrival -> card_delivery_estimate: 4 times
- balance_not_updated_after_bank_transfer -> transfer_timing: 4 times
- balance_not_updated_after_bank_transfer -> transfer_not_received_by_recipient: 4 times
- wrong_amount_of_cash_received -> declined_cash_withdrawal: 3 times

## Weakest 5 intents (lowest F1)

- balance_not_updated_after_bank_transfer: F1 = 0.753
- pending_transfer: F1 = 0.810
- declined_transfer: F1 = 0.817
- transfer_not_received_by_recipient: F1 = 0.837
- top_up_failed: F1 = 0.840
