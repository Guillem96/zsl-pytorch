FILES="AwA-base.tar.bz2 AwA-features-cq.tar.bz2 AwA-features-lss.tar.bz2"
FILES="$FILES AwA-features-phog.tar.bz2 AwA-features-sift.tar.bz2"
FILES="$FILES AwA-features-rgsift.tar.bz2 AwA-features-surf.tar.bz2"

for f in $FILES; do
    rm $f
    echo "Downloading $f"
    wget $f
    echo "Uncompressing $f"
    tar xjf $f
done

