from app.dataset import download_dataset


def main():
    path = download_dataset()
    print("Path to dataset files:", path)


if __name__ == '__main__':
    main()
