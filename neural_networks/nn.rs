use candle_core::*;
use candle_nn::*;
use anyhow::Result;
use std::fs::File;
use std::io::Read;

const VOTE_DIM: usize = 7;
const RESULTS: usize = 1;
const EPOCHS: usize = 100;
const LAYER1_OUT_SIZE: usize = 32;
const LAYER2_OUT_SIZE: usize = 64;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_values: Tensor,
    pub train_results: Tensor,
    pub test_values: Tensor,
    pub test_results: Tensor,
}

struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let train_results = m.train_results.to_device(dev)?;
    let train_values = m.train_values.to_device(dev)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = MultiLevelPerceptron::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let test_values = m.test_values.to_device(dev)?;
    let test_results = m.test_results.to_device(dev)?;
    let mut final_accuracy: f32 = 0.0;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_values)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        let loss = loss::nll(&log_sm, &train_results)?;
        sgd.backward_step(&loss)?;
        let test_logits = model.forward(&test_values)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_results)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_results.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                 loss.to_scalar::<f32>()?,
                 final_accuracy
        );
        if final_accuracy >= 90.0 {
            break;
        }
    }
    if final_accuracy < 90.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}

fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    // Leggi i dati di addestramento e test da file
    let mut data_file = File::open("banana_quality_3.csv")?;
    let mut data = String::new();
    data_file.read_to_string(&mut data)?;
    let mut lines = data.lines();
    let mut train_values: Vec<u32> = Vec::new();
    let mut train_results: Vec<u32> = Vec::new();
    let mut test_values: Vec<u32> = Vec::new();
    let mut test_results: Vec<u32> = Vec::new();

    let mut l_values: Vec<u32> = Vec::new();
    let mut l_results: Vec<u32> = Vec::new();

    // Prendi dalla riga 1 in poi
    for line in lines.skip(1) {
        let mut values = line.split(",");
        // Size,Weight,Sweetness,Softness,HarvestTime,Ripeness,Acidity,Quality
        // Saraà una lista di interi unsigned
        let size: f32 = values.next().unwrap().parse().unwrap();
        let weight: f32 = values.next().unwrap().parse().unwrap();
        let sweetness: f32 = values.next().unwrap().parse().unwrap();
        let softness: f32 = values.next().unwrap().parse().unwrap();
        let harvest_time: f32 = values.next().unwrap().parse().unwrap();
        let ripeness: f32 = values.next().unwrap().parse().unwrap();
        let acidity: f32 = values.next().unwrap().parse().unwrap();
        let quality: f32 = values.next().unwrap().parse().unwrap(); 
        
        let values = vec!
        [
            size as u32,
            weight as u32,
            sweetness as u32,
            softness as u32,
            harvest_time as u32,
            ripeness as u32,
            acidity as u32,
        ];

        l_values.extend(values.clone());
        l_results.push(quality as u32);
    }
    //Dividi i dati in training e test con un rapporto 80/20
    for i in 0..8000 {
        if i % 5 == 0 {
            for j in 0..VOTE_DIM {
                test_values.push(l_values[i * VOTE_DIM + j]);
            }
            test_results.push(l_results[i]);
        } else {
            for j in 0..VOTE_DIM {
                train_values.push(l_values[i * VOTE_DIM + j]);
            }
            train_results.push(l_results[i]);
        }
    }


    println!("train_values: {:?}", train_values.len());
    println!("train_results: {:?}", train_results.len());
    println!("test_values: {:?}", test_values.len());
    println!("test_results: {:?}", test_results.len());
    



    let m = Dataset {
        train_values: Tensor::from_vec(train_values.clone(), (train_values.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?,
        train_results: Tensor::from_vec(train_results, train_values.len() / VOTE_DIM, &dev)?,
        test_values: Tensor::from_vec(test_values.clone(), (test_values.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?,
        test_results: Tensor::from_vec(test_results, test_values.len() / VOTE_DIM, &dev)?,
    };

    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            },
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        }

    }

    Ok(())
}
