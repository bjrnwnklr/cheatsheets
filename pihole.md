# Setting up a Raspberry Pi as a Pi-hole for the Ubiquiti network

# Articles

[[Step-By-Step Tutorial/Guide] Raspberry Pi with UniFi Controller and Pi-hole from scratch (headless)](https://community.ui.com/questions/Step-By-Step-Tutorial-Guide-Raspberry-Pi-with-UniFi-Controller-and-Pi-hole-from-scratch-headless/e8a24143-bfb8-4a61-973d-0b55320101dc)
[Mit dem Pi-hole einen Werbeblocker für das gesamte lokale Netz einrichten und dank UniFi Access Points nie mehr WiFi-Probleme [Tutorial]](https://medium.com/@natterstefan/mit-dem-pi-hole-einen-werbeblocker-f%C3%BCr-das-gesamte-lokale-netz-einrichten-und-mit-unifi-access-5f087a13ff5a)

# 1. Setting up the Raspberry Pi

## Install Raspbian OS

[Installing operating sytem images using Raspberry Pi Imager](https://www.raspberrypi.org/documentation/installation/installing-images/README.md)

- Create a file called `ssh` on the SD card to enable SSH connections on the RPi.
- Insert the card into the RPi, plug into network and connect power to start the RPi
- Find the IP address of the RPi by going to the Unifi USG (192.168.1.1), log in via the account and check the DHCP table for the IP address of the `raspberrypi` client

## Connecting to the Pi and update

- SSH to the IP of the Pi and log in using `pi:raspberry` (default user / password.)
- change the default password

```console
$ ssh pi:raspberry@<IP-Address>
$ passwd # to change password
```

- Assign fixed IP address:
    - In the Unifi Controller app (cloud key), find the Pi under Clients and go into Settings > Network to set a fixed IP address
    - Release the IP address on the Pi using `sudo dhclient -v -r` (which also kills the network interface so you need to reconnect with SSH to the new IP)

- Update all software packages

```console
$ sudo apt-get update -y
$ sudo apt-get upgrade -y
$ sudo apt-get autoremove
$ sudo apt-get autoclean
```

- Turn of swapping (default 100MB) - not required for a server

As per [Pi-hole](https://discourse.pi-hole.net/t/pi-hole-sd-card-schonung/12727/24) - turn off swapping on Pi-holes as it is really not required.

**For now, I keep the swapping enabled and at 100MB default size**

More info on [swapping](https://www.elektronik-kompendium.de/sites/raspberry-pi/2002131.htm).

Check status of swapping:

```console
$ sudo service dphys-swapfile status
```

Turn swapping off and on

```console
$ sudo swapoff -a
$ sudo swapon -a
```

Check usage of swap

```console
$ free -h
```

Edit swap file size

```console
$ sudo nano /etc/dphys-swapfile
```

Completely disable swapping

```console
$ sudo service dphys-swapfile stop # stop service
$ free # check if swapping is off
$ sudo systemctl disable dphys-swapfile # deactive swap service
$ sudo apt-get purge dphys-swapfile # remove swap fully
```

Re-install swapping (edit `etc/dphys-swapfile` to set a value for swap size)

```console
$ sudo apt-get install dphys-swapfile
```

## Installing Pi-Hole

[Install Pi-Hole](https://github.com/pi-hole/pi-hole/#one-step-automated-install):

```console
$ curl -sSL https://install.pi-hole.net | bash
```

- Select one of the standard DNS providers (Google, OpenDNS or Cloudflare)
- Keep all adblocking providers enabled
- Select IP4 and IP6
- Install web interface and LightHppd web server
- Privacy mode for FTL: 0 Show everything

Admin web interface can be found at http://<IP of RPi>/admin or http://pi.hole/admin

Password is 9yGbRcWN - change this

Change password:

```console
$ pihole -a -p
```

## Configure Pi-Hole

- Log in to the admin interface at http://<IP of RPi>/admin, select Login and login with the password.
- Under Settings > DNS, 
    - select the following Upstream DNS servers
        - Google (ECS)
        - Cloudflare
    - Turn on "Use DNSSEC"
    - Turn on "Use Conditional Forwarding"
        - Local network in CIDR notation: 192.168.1.1/24
        - IP address of your DHCP server (router): 192.168.1.1
        - Local domain name: localdomain

## Configure Unifi network

- In Settings > Networks > Local Networks > Edit
    - DHCP Name Server: Manual, enter IP address of RPi under DNS Server 1

- In USG, under Services > DHCP > DHCP Server, be sure `Register client hostname from DHCP requests in USG DNS forwarder` is `On`

- Configure Upstream DNS servers for USG:
    - In USG > Internet > WAN Networks > Edit
        - DNS Servers: set to 1.1.1.1 (Cloudflare) and 8.8.8.8 (Google)


## Backup

Backup the settings of Pi-Hole either via the admin web interface (Settings > Teleporter) or via SSH:

```console
$ pihole -a teleporter
```

## Shutdown the Raspberry Pi including Pi-Hole

The Raspberry Pi needs to be gracefully shutdown or you risk corruption.

Use the following commands to shut down:

```console
$ sudo shutdown -h now
$ sudo poweroff
```

To reboot

```console
$ sudo shutdown -r now
$ sudo reboot
```