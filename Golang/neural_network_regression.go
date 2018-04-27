package main

import (
	"gonum.org/v1/gonum/mat" //has to follow this directory
	"gonum.org/v1/gonum/stat"

	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

const HIDDEN_LAYER_SIZE int = 10
const ALPHA float64 = 0.03
const LAMBDA float64 = 0.01
const NUM_ITERS int = 1000

func main() {
	fName := "D:\\Ziming\\2018\\0_MachineLearningTraining\\Summary\\data\\fitStudy2.txt"
	op, err := os.Open(fName)
	if err != nil {
		log.Fatal(err)
	}
	defer op.Close()
	scanner := bufio.NewScanner(op)

	//read all data into dataMat
	cnt := 0
	colNum := 0 //number of columns (including y)
	fList := make([]float64, 0)
	for scanner.Scan() {
		tLine := scanner.Text()
		tempList := strings.Split(tLine, "\t")
		if cnt > 0 {
			for _, v := range tempList {
				f, err := strconv.ParseFloat(v, 64)
				if err != nil {
					log.Fatal(err)
				}
				fList = append(fList, f)
			}
		} else {
			colNum = len(tempList)
		}
		cnt += 1
	}
	rowNum := cnt - 1

	//separate X and y from dataMat
	m := rowNum     //#samples
	n := colNum - 1 //#features
	dataMat := mat.NewDense(rowNum, colNum, fList)
	X := mat.NewDense(m, n, nil)
	for j := 0; j < n; j++ {
		X.SetCol(j, mat.Col(nil, j, dataMat))
	}
	y := mat.NewDense(m, 1, mat.Col(nil, n, dataMat))

	//initialize thetas
	theta1, theta2 := thetaInitialization(n)
	_, _, _, _ = X, y, theta1, theta2

	//---------------debug--------------------------
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < n+1; j++ {
			theta1.Set(i, j, 0.1)
		}
	}
	for i := 0; i < HIDDEN_LAYER_SIZE+1; i++ {
		theta2.Set(0, i, 0.1)
	}
	//----------------------------------------------

	//normalization
	muX, stdX := make([]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		muX[i], stdX[i] = stat.MeanStdDev(mat.Col(nil, i, X), nil)
	}
	XNorm := mat.NewDense(m, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			XNorm.Set(j, i, (X.At(j, i)-muX[i])/stdX[i])
		}
	}

	muY, stdY := stat.MeanStdDev(mat.Col(nil, 0, y), nil)
	yNorm := mat.NewDense(m, 1, nil)
	for j := 0; j < m; j++ {
		yNorm.Set(j, 0, (y.At(j, 0)-muY)/stdY)
	}

	//gradient descent
	ones := make([]float64, m)
	for i, _ := range ones {
		ones[i] = 1
	}
	for i := 0; i < NUM_ITERS; i++ {
		//forward propagation
		a1 := mat.NewDense(m, n+1, nil)
		a1.SetCol(0, ones)
		for j := 0; j < n; j++ {
			a1.SetCol(j+1, mat.Col(nil, j, XNorm))
		}

		var a2Temp mat.Dense
		a2Temp.Mul(theta1, a1.T())

		for j := 0; j < HIDDEN_LAYER_SIZE; j++ {
			for k := 0; k < m; k++ {
				a2Temp.Set(j, k, math.Tanh(a2Temp.At(j, k)))
			}
		}
		onesMat := mat.NewDense(1, m, ones)
		a2 := mat.NewDense(HIDDEN_LAYER_SIZE+1, m, nil)
		a2.Stack(onesMat, &a2Temp)

		var h mat.Dense
		h.Mul(theta2, a2) //same dimention as y and yNorm (mx1)

		var d, d2 mat.Dense
		d.Sub(&h, yNorm.T())
		d2.MulElem(&d, &d)
		J := 1 / float64(2*m) * mat.Sum(&d2)

		theta1_reg := theta1.Slice(0, HIDDEN_LAYER_SIZE, 1, n+1)
		theta2_reg := theta2.Slice(0, 1, 1, HIDDEN_LAYER_SIZE+1)
		var t1, t2 mat.Dense
		t1.MulElem(theta1_reg, theta1_reg)
		t2.MulElem(theta2_reg, theta2_reg)
		reg := LAMBDA / float64(2*m) * (mat.Sum(&t1) + mat.Sum(&t2))
		J += reg

		if i%10 == 0 {
			fmt.Println(J)
		}

		//backward propagation
		var delta3, Delta2, Delta1 mat.Dense
		delta3.Sub(&h, yNorm.T())
		var tm1, tm2, tm3, tm4 mat.Dense
		tm1.Mul(theta2.T(), &delta3)
		tm2.MulElem(a2, a2)
		tm3.MulElem(&tm1, &tm2)
		tm4.Sub(&tm1, &tm3)
		delta2 := tm4.Slice(1, HIDDEN_LAYER_SIZE+1, 0, m)

		Delta2.Mul(&delta3, a2.T())
		Delta1.Mul(delta2, a1)

		var theta1_zeroBias, theta2_zeroBias mat.Dense

		theta2_zeroBias.Clone(theta2)
		theta2_zeroBias.Set(0, 0, 0)
		theta1_zeroBias.Clone(theta1)

		for k := 0; k < HIDDEN_LAYER_SIZE; k++ {
			theta1_zeroBias.Set(k, 0, 0)
		}

		Delta2.Scale(1/float64(m), &Delta2)
		Delta1.Scale(1/float64(m), &Delta1)
		theta2_zeroBias.Scale(LAMBDA/float64(m), &theta2_zeroBias)
		theta1_zeroBias.Scale(LAMBDA/float64(m), &theta1_zeroBias)
		var theta2_grad, theta1_grad mat.Dense
		theta2_grad.Add(&Delta2, &theta2_zeroBias)
		theta1_grad.Add(&Delta1, &theta1_zeroBias)

		theta1_grad.Scale(ALPHA, &theta1_grad)
		theta2_grad.Scale(ALPHA, &theta2_grad)

		theta1.Sub(theta1, &theta1_grad)
		theta2.Sub(theta2, &theta2_grad)
	}

	//get prediction (be sure to un-normalize h)
	a1 := mat.NewDense(m, n+1, nil)
	a1.SetCol(0, ones)
	for j := 0; j < n; j++ {
		a1.SetCol(j+1, mat.Col(nil, j, XNorm))
	}
	var a2Temp mat.Dense
	a2Temp.Mul(theta1, a1.T())
	for j := 0; j < HIDDEN_LAYER_SIZE; j++ {
		for k := 0; k < m; k++ {
			a2Temp.Set(j, k, math.Tanh(a2Temp.At(j, k)))
		}
	}
	onesMat := mat.NewDense(1, m, ones)
	a2 := mat.NewDense(HIDDEN_LAYER_SIZE+1, m, nil)
	a2.Stack(onesMat, &a2Temp)
	var h mat.Dense
	h.Mul(theta2, a2)
	for j := 0; j < m; j++ {
		h.Set(0, j, h.At(0, j)*stdY+muY)
	}
	fmt.Println(h)
}

func thetaInitialization(n int) (theta1 *mat.Dense, theta2 *mat.Dense) {
	epsi_1 := math.Sqrt(6 / float64(n+HIDDEN_LAYER_SIZE))
	epsi_2 := math.Sqrt(6 / float64(HIDDEN_LAYER_SIZE+1))
	theta1 = mat.NewDense(HIDDEN_LAYER_SIZE, n+1, nil)
	theta2 = mat.NewDense(1, HIDDEN_LAYER_SIZE+1, nil)

	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < n+1; j++ {
			t := rand.Float64()
			theta1.Set(i, j, t*2*epsi_1-epsi_1)
		}
	}
	for j := 0; j < HIDDEN_LAYER_SIZE+1; j++ {
		t := rand.Float64()
		theta2.Set(0, j, t*2*epsi_2-epsi_2)
	}
	return
}
