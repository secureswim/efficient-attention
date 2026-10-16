# Storage operations

## Snapshot policy

The snapshot interval is HOURLY. The storage service records a recovery point after each completed snapshot.

### Recovery check

Once a day, restore a sample snapshot in the test environment and verify its checksum.

## Log policy

Retain storage logs for FOURTEEN days. Rotate the log files at midnight to keep the active directory small.

- Archive yesterday's log.
- Remove expired archives.
