FILES="AwA2-base.zip AwA2-features.zip"

mkdir -p AwA2

for f in $FILES; do
    rm -f $f
    echo "Downloading $f"
    wget "http://cvml.ist.ac.at/AwA2/$f"

    echo "Uncompressing $f"
    unzip $f -d AwA2
    rm $f
done

