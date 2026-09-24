# Running recoup on an old Linux laptop

Two things run on the laptop: the **dashboard builder** (nightly, ~2–3 min) and
a **static web server** for the console page. Everything else in
`deploy/systemd/` (retrain / plan / gate) needs the `recoup-ops` entry point and
a pinned Bachs adapter, which don't exist yet — leave those units installed but
not enabled until then.

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

**Do not** `enable recoup.target` yet — it also pulls in retrain/plan/gate,
which will fail until `recoup-ops` exists.

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

Change `simulate()` to `from_payments()` in `export_dashboard.py`, remove
`PrivateNetwork=yes` from `recoup-dashboard.service`, add the API key with
`systemd-creds` as described in `systemd/README.md`, then write `recoup-ops`
and enable `recoup.target`. Nothing about the web server or the page changes.
