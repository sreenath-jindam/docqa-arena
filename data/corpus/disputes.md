# Disputes

A dispute, also called a chargeback, is created when a cardholder asks their bank to reverse a charge. The bank pulls the funds from your account while it investigates, and you have a fixed window to submit evidence that the charge was legitimate.

## The dispute lifecycle

When a dispute is opened, the disputed amount and a separate dispute fee are immediately withdrawn from your balance. The Dispute object is created with a status of `needs_response` and an `evidence_details.due_by` timestamp.

Submitting evidence moves the dispute to `under_review`. The issuing bank then takes anywhere from 60 to 75 days to reach a decision. A dispute you win moves to `won` and the disputed amount is returned to your balance; the dispute fee is not returned. A dispute you lose moves to `lost` and nothing is returned.

If you do not respond before the due date, the dispute closes automatically in the cardholder's favour and moves to `lost`.

## Evidence

Evidence is submitted as a structured object rather than free text, because issuing banks score specific fields. The fields that matter most depend on the dispute reason. For a `product_not_received` dispute, shipping documentation with a tracking number and a delivery confirmation carries the most weight. For a `fraudulent` dispute, the customer's IP address, billing address match, and any record of prior undisputed purchases from the same customer are what the bank looks at.

Evidence can be updated any number of times before it is submitted, but submission is final. Once you submit, the evidence object becomes read-only.

## Dispute fees

The dispute fee is charged the moment the dispute is created and is not refunded regardless of outcome. This is a bank cost, not a service charge, and it applies even to disputes you win. The fee amount varies by country and currency.

## Early fraud warnings

Some card networks send a fraud signal before a formal dispute is filed. An early fraud warning indicates the issuer has flagged the charge as fraudulent but the cardholder has not yet filed a chargeback. Refunding the charge at this stage usually prevents the dispute from being filed at all, which avoids the dispute fee and keeps the charge off your dispute rate.

## Dispute rate

Card networks monitor the ratio of disputes to total charges. Exceeding roughly one percent puts an account into a network monitoring programme, which carries monthly fines and can eventually result in the account losing card acceptance. Refunds do not count toward the dispute rate, which is why refunding an early fraud warning is almost always cheaper than fighting the resulting chargeback.

## Webhooks

`charge.dispute.created` fires when the dispute opens, `charge.dispute.updated` fires when evidence is submitted or the status changes, and `charge.dispute.closed` fires on the final decision. The `closed` event carries the final status in its payload.
