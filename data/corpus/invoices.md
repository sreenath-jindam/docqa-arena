# Invoices and subscriptions

An Invoice is a statement of amounts owed by a customer. It can be created directly for a one-off bill, or generated automatically at the end of each billing period by a Subscription.

## Invoice lifecycle

An invoice starts as a `draft`. While it is a draft it can be edited freely: line items added or removed, the customer changed, discounts applied. Drafts are not visible to the customer and no payment is attempted.

Finalising an invoice moves it to `open` and freezes its line items. An open invoice has a hosted payment page and, depending on your settings, triggers an automatic collection attempt or an email to the customer.

An invoice that is paid moves to `paid`. One that is written off moves to `uncollectible`. One that is cancelled before payment moves to `void`. A voided invoice is preserved for your records but is treated as though it never existed for revenue purposes.

Invoices are finalised automatically one hour after creation unless you set `auto_advance` to false, which keeps the invoice in draft indefinitely and hands control of finalisation to your own code.

## Subscription billing periods

A subscription generates an invoice at the start of each billing period, covering that period in advance. The invoice is created a few hours before the period begins, which gives failed payments time to retry before service is interrupted.

Changing a subscription mid-period produces a proration. Two line items are added to the next invoice: a credit for the unused portion of the old price, and a charge for the remaining portion of the new one. Set `proration_behavior` to `none` to change the price with no adjustment for the period already elapsed.

## Failed payments and dunning

When an invoice payment fails, the subscription moves to `past_due` and the retry schedule begins. The default schedule retries after 3 days, then 5 days, then 7 days. After the final retry, the subscription moves to whatever you configured as the terminal state: `canceled`, `unpaid`, or left in `past_due`.

A subscription in `unpaid` keeps generating invoices but does not attempt to collect them. A subscription in `canceled` stops generating invoices entirely and cannot be reactivated — you must create a new subscription.

## Trials

A subscription with a trial period does not generate a payable invoice until the trial ends. A zero-amount invoice is still created at the start of the trial so that the billing period is recorded. Trials can be extended while active by updating `trial_end`, but a trial cannot be added to a subscription after it has started billing.

## Credit notes

A credit note reduces the amount owed on an invoice that has already been finalised. Because a finalised invoice is immutable, this is the only way to correct an overcharge after finalisation. A credit note on a paid invoice can either refund the customer or add the amount to their customer balance, applied automatically against their next invoice.
