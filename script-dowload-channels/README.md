# Dowload Channels

This script creates JSON files with data from Telegram channels. It allows you to choose a channel or multiple channels from a text file. The script automatically creates *output/data* folders for file storage. 

**Requirements:**

- [Python 3.x](https://www.python.org/)
- [Telegram API credentials](https://my.telegram.org/auth?to=apps)
    - Telegram account
    - App `api_id`
    - App `api_hash`
- [Telethon](https://docs.telethon.dev/en/stable/)
- [Pandas](https://pandas.pydata.org/)

## `main.py` 
### Options
- `--telegram-channel` Choose a Telegram Channel to download data.
- `--batch-file` Use a file with a list of Telegram Channels (one per line).
- `--limit-download-to-channel-metadata` Collect only channel metadata (default = False).
- `--output, -o` Choose a folder for data storage. Default is `./output/data`.
- `--min-id` Set the starting ID for downloading new posts.

### Output 📂
```
├──🗂 output
| └──🗂 data
| └──🗂 <channel_name>
| ├── <channel_name>.json
| ├── <channel_name>_messages.json
| ├── chats.txt
| ├── collected_chats.csv
| ├── collected_chats.xlsx
| ├── counter.csv
| ├── user_exceptions.txt
| └── msgs_dataset.csv
```
### Use

```
python main.py --telegram-channel channelname
```

**Output:**
- Channel files: chats.txt, collected_chats.csv, user_exceptions.txt, counter.csv.
- Folder `<channel_name>` with two JSON files (channel's metadata and posts).

**Note:** Data is saved in the specified directory.

---
Contact Email: malbaposse@mail.utdt.edu 
Credits to **estebanpdl** which repository I used for doing mine. 🙌
---