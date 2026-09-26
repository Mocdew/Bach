# Running recoup on an old Linux laptop

Two things run on the laptop: the **dashboard builder** (nightly, ~2–3 min) and
a **static web server** for the console page. The batch jobs in
`deploy/systemd/` (retrain / plan / gate) run `recoup-ops` against a dataset
directory; until a Bachs fetcher exists that is a synthetic one — step 5b sets
it up, and it is optional.

What you see on your main laptop at the end: `http://<laptop-ip>:8080/`, the
same console as the Claude artifact, refreshed every night from the server.

---

## 0. Before you start (on the old laptop)

Install a minimal server OS — Ubuntu Server 24.04 or Debian 12, no desktop.
During install, create a user (say `ops`) and tick "install OpenSSH server".
Plug the laptop into mains and connect it to your home network (ethernet if it
has a port; wifi is fine).

## 1. Make a laptop behave like a server

```bash
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/; s/^#\?HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf && sudo systemctl restart systemd-logind
```

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Close the lid; it should stay on. Then check the disk — this is the part of an
old laptop most likely to die:

```bash
sudo apt install -y smartmontools && sudo smartctl -H /dev/sda
```

`PASSED` is what you want. If it says anything else, put a different drive in
before trusting it with anything.

## 2. Find its address and make it stable

```bash
hostname -I
```

Note the address (e.g. `192.168.1.42`). In your router's admin page, reserve
that IP for the laptop's MAC address ("DHCP reservation" / "static lease") so it
doesn't change after a reboot. Alternatively skip the router and use Tailscale
(step 7), which gives it a permanent name.

## 3. Install Python and the code

```bash
sudo apt install -y python3 python3-venv git
```

```bash
sudo mkdir -p /opt/recoup && sudo chown $USER /opt/recoup && git clone <your-repo-url> /opt/recoup/src
```

(If the repo isn't pushed anywhere yet, copy it from your main laptop instead —
see step 6 for the `scp` direction, reversed.)

```bash
python3 -m venv /opt/recoup/venv && /opt/recoup/venv/bin/pip install --upgrade pip && /opt/recoup/venv/bin/pip install /opt/recoup/src
```

```bash
cp -r /opt/recoup/src/deploy/dashboard /opt/recoup/dashboard
```

Sanity check — this is the exact job the timer will run, so if it works here
it works under systemd:

```bash
mkdir -p /tmp/www && /opt/recoup/venv/bin/python /opt/recoup/dashboard/export_dashboard.py /tmp/www/dashboard.json && ls -la /tmp/www
```

Expect 2–3 minutes and a ~40 KB `dashboard.json`.

## 4. Install the systemd units

```bash
cd /opt/recoup/src/deploy/systemd && sudo install -Dm644 recoup.slice recoup.target recoup-*.service recoup-*.timer -t /etc/systemd/system/
```

```bash
for u in retrain plan gate model-present; do sudo install -Dm644 /opt/recoup/src/deploy/systemd/recoup-common.conf /etc/systemd/system/recoup-$u.service.d/00-common.conf; done
```

Put the page where the web server serves from. `StateDirectory=recoup` with
`DynamicUser=` lives under `/var/lib/private/recoup`, so create it via a unit
run rather than by hand — the first dashboard build does it:

```bash
sudo systemctl daemon-reload && sudo systemctl start recoup-dashboard.service
```

That blocks for the 2–3 minute build. Then drop the page next to the JSON:

```bash
sudo install -Dm644 /opt/recoup/dashboard/index.html /var/lib/private/recoup/www/index.html
```

## 5. Turn on the web server and the nightly rebuild

```bash
sudo systemctl enable --now recoup-web.service recoup-dashboard.timer
```

```bash
systemctl status recoup-web.service --no-pager && systemctl list-timers 'recoup-*' --no-pager
```

On the laptop itself: `curl -s localhost:8080/dashboard.json | head -c 200`
should print JSON.

## 5b. (Optional) run the batch jobs on synthetic data

`recoup.target` pulls in retrain / plan / gate. They need a dataset in
`RECOUP_DATA=/var/lib/recoup/data`; make a synthetic one as the same dynamic
user the jobs run as, so the ownership is right:

```bash
sudo systemd-run --wait --pipe -p DynamicUser=yes -p StateDirectory=recoup -p UMask=0077 /opt/recoup/venv/bin/recoup-ops synth --out /var/lib/recoup/data
```

```bash
sudo systemctl start recoup-retrain.service && sudo systemctl enable --now recoup.target
```

The synthetic world's clock does not move on its own: `plan` will decide the
open invoices once, and then find nothing new. To step it forward a day
(executing the queued retries against the simulator), run the same
`systemd-run` line with `advance --data /var/lib/recoup/data --queue
/var/lib/recoup/queue --hours 24`. With real data none of this applies — the
clock is the wall clock.

## 5c. (Optional) the live console

The static page shows last night's snapshot. `recoup-console.service` serves
the live console (API + UI, see `web/README.md`) on port 8081 over the same
state the jobs write. Build the UI on your main laptop — the server needs no
Node.js — then copy it over:

```bash
cd web && npm ci && npm run build && scp -r dist ops@192.168.1.42:/tmp/recoup-web
```

On the server:

```bash
/opt/recoup/venv/bin/pip install '/opt/recoup/src[api]' && sudo rm -rf /opt/recoup/web && sudo mv /tmp/recoup-web /opt/recoup/web
```

```bash
sudo install -Dm644 /opt/recoup/src/deploy/systemd/recoup-console.service -t /etc/systemd/system/ && sudo install -Dm644 /opt/recoup/src/deploy/systemd/recoup-common.conf /etc/systemd/system/recoup-console.service.d/00-common.conf && sudo systemctl daemon-reload && sudo systemctl enable --now recoup-console.service
```

Then `http://192.168.1.42:8081/`. Allow the port the same way as step 6.

## 6. Open it on your main laptop

Same wifi/LAN: open **`http://192.168.1.42:8080/`** (your address from step 2)
in any browser. The pill at the top reads "live · simulator …" instead of
"simulated data" — that's the page confirming it fetched `dashboard.json`
from the server rather than using its embedded copy.

If it doesn't load, the laptop has a firewall on:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp
```

(Adjust the subnet to yours. This allows the LAN only, never the internet.)

To pull the raw results to Windows for your own analysis, from PowerShell:

```powershell
scp ops@192.168.1.42:/var/lib/private/recoup/www/dashboard.json $HOME\Downloads\
```

## 7. Reaching it from outside the house (optional)

Don't port-forward 8080 on the router — a page with a payment-recovery
dashboard on it is not something to leave on the open internet, and the
`http.server` behind it has no auth. Use Tailscale on both laptops instead:

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

Install the Windows client, sign in with the same account, and the server is
reachable at `http://<laptop-name>:8080/` from anywhere, encrypted, with no
open ports. The `ufw` rule from step 6 needs `tailscale0` allowed too:
`sudo ufw allow in on tailscale0 to any port 8080`.

## 8. Day-to-day

| want to… | command on the server |
|---|---|
| rebuild the dashboard now | `sudo systemctl start recoup-dashboard.service` |
| see why last night's build failed | `journalctl -u recoup-dashboard -n 50 --no-pager` |
| see the web server log | `journalctl -u recoup-web -f` |
| update the code | `cd /opt/recoup/src && git pull && /opt/recoup/venv/bin/pip install . && cp deploy/dashboard/* /opt/recoup/dashboard/ && sudo install -m644 deploy/dashboard/index.html /var/lib/private/recoup/www/` |
| back up the results | `rsync -a /var/lib/private/recoup/ <somewhere else>/` — nightly, via a cron/timer of your own; it's a few MB |

## When the Bachs adapter is real

Write the fetcher that fills `RECOUP_DATA` from the API (and the executor that
turns `queue/*.jsonl` into retry calls), change `simulate()` to
`recoup.store.load_dataset()` in `export_dashboard.py`, remove
`PrivateNetwork=yes` from `recoup-dashboard.service`, and add the API key with
`systemd-creds` as described in `systemd/README.md`. Nothing about the web
server, the page or the `recoup-ops` jobs changes.
