# Refunds

A refund returns money from your account balance to a customer who was previously charged. Refunds are created against a specific charge and can never exceed the amount of that charge.

## Creating a refund

To refund a charge in full, create a Refund object with only the charge identifier. To refund part of a charge, pass an `amount` in the smallest currency unit — 1250 for $12.50. You may issue multiple partial refunds against the same charge as long as their combined total does not exceed the original charge amount. Attempting to exceed it returns a `charge_already_refunded` error.

A refund cannot be cancelled once it has been created. If you refund by mistake, the only remedy is to charge the customer again.

## Settlement timing

A refund is submitted to the customer's bank immediately, but the funds do not appear on their statement right away. Card refunds typically take five to ten business days to settle, and the exact timing is controlled by the issuing bank rather than by us. Bank debit refunds settle more slowly, usually within ten to fifteen business days.

During this window the Refund object has a status of `pending`. It moves to `succeeded` once the issuer confirms the credit, or to `failed` if the issuer rejects it — most commonly because the customer's card was closed between the charge and the refund.

## Fees

The processing fee from the original charge is not returned when you refund. This is true for both full and partial refunds. For a $100 charge with a $3.20 fee, a full refund returns $100 to the customer and leaves your balance $3.20 lower than before the charge existed.

Application fees behave differently. When a charge that carried an application fee is refunded, the application fee is refunded proportionally and automatically, unless you set `refund_application_fee` to false at the time of the refund.

## Refunds and negative balances

If your available balance is lower than the refund amount, the refund still proceeds and your balance goes negative. A negative balance is settled by your next incoming payments, or by a debit from your bank account if no payments arrive within seven days.

## Refunds on disputed charges

A charge that is under dispute cannot be refunded. The dispute process determines where the money goes, and issuing a refund on top of it would move the same funds twice. Withdraw the dispute response first, or wait for the dispute to close.

## Webhooks

Two events are emitted for every refund. `refund.created` fires when the refund is submitted, and `refund.updated` fires when its status changes — most importantly on the transition from `pending` to `succeeded` or `failed`. Rely on `refund.updated`, not `refund.created`, if your application needs to know the money actually reached the customer.
