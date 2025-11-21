# WP AI Impact Inserter

This repository contains a Python script that connects to a WordPress site and adds a dynamically generated expert section to selected posts via the WordPress REST API. The new section extends the existing article with up to date context, for example an extra block with a heading and several paragraphs related to a cross topic such as `[CROSS_TOPIC_HEADING]` that you define for your own use case.

The script uses the DeepSeek chat completion API to generate French language text that fits the topic of each article and is inserted as an additional section at the end of the post.

The general idea behind this project:

> Large Language Models (LLMs) make it possible to extend existing content with new developments and more specific angles. For websites with a large amount of niche content this can be a practical way to keep articles current without rewriting them from scratch.

The code contains no hard coded credentials. All secrets are read from environment variables.

## Features

- Connects to WordPress via the REST API  
- Supports basic auth and cookie plus nonce auth  
- Filters posts by status, length, and title pattern (`only_title_contains`)  
- Checks for existing cross topic sections to avoid duplicates  
- Uses DeepSeek to  
  - derive a suitable H2 heading about a cross topic (for example `[CROSS_TOPIC_HEADING]`)  
  - select a generic expert role label (for example `expert du sujet`)  
  - generate a French opening question  
  - generate three short French paragraphs  
- Appends the generated expert block at the end of the post content  
- Can run in dry run mode and supports automatic batching  

## Requirements

- Python 3.10 or newer  
- Packages:  
  - `requests` (which brings `urllib3`)  

You can install dependencies with:

```bash
pip install requests
