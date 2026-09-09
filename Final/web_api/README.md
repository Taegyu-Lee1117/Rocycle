# Rocycle UI/DB API

This server serves `web_ui` and connects it to the local PostgreSQL database.
The browser never connects to PostgreSQL directly.

## Ubuntu setup

```bash
cd ~/Rocycle/Final
cp .env.example .env
nano .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r web_api/requirements.txt
python3 -m web_api.main
```

Open on the Ubuntu PC:

```text
http://127.0.0.1:8000
```

Open from another PC on the same network:

```text
http://172.24.0.31:8000
```

Health check and API documentation:

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/docs
```

The first API startup applies the idempotent schema in
`DB/rocycle_project_schema.sql`. The PostgreSQL user must own the tables or
have permission to create and alter them.

## Connect completed ROS2 cycles

Run the read-only DB bridge in another sourced ROS2 terminal:

```bash
cd ~/Rocycle/Final
colcon build --symlink-install
source install/setup.bash
ros2 run rocycle_robot db_bridge_node
```

The bridge subscribes to `/ui/state` and sends a record only after `last.dest`
is present. A stable `external_id` makes repeated state messages idempotent.
It never publishes a robot or conveyor command.

## API-only test record

```bash
curl -X POST http://127.0.0.1:8000/api/processing \
  -H 'Content-Type: application/json' \
  -d '{"external_id":"manual-test-1","predicted_class":"plastic","confidence":0.72,"destination":"review_bin","status":"review_pending","review_reason":"weight_exceeded"}'
```

Then open the review list in the UI. Choosing a final class and pressing the
completion button updates `processing_log`, inserts a `review_completed`
event, decreases the pending count, and refreshes the dashboard.
