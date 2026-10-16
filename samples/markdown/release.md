# Release handbook

## Production rollout

The production rollout uses BLUE as its rollback target. Keep that target ready until the new deployment passes validation.

### Operator steps

- Promote the candidate release.
- Verify the health endpoint.
- If validation fails, return traffic to BLUE.

## Canary checks

The canary runs in the EAST region. Check the service metrics for ten minutes before wider rollout.

```text
canary-region=EAST
```
