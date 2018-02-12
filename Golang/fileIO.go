package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

func main() {
	fmt.Print("enter a directory: ")
	inBuf := bufio.NewReader(os.Stdin)
	inDir, _ := inBuf.ReadString('\n')
	inDir = inDir[0 : len(inDir)-2]
	files, _ := ioutil.ReadDir(inDir)

	for _, f := range files {
		fName := inDir + "\\" + f.Name()
		op, err := os.Open(fName)
		if err != nil {
			log.Fatal(err)
		}
		defer op.Close()
		scanner := bufio.NewScanner(op)

		for scanner.Scan() {
			tLine := scanner.Text()
			fmt.Println(tLine)
		}
	}

	wp, err := os.Create("C:\\Test\\out.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer wp.Close()
	wp.WriteString("lalalalala" + "\t" + "\n")
}
