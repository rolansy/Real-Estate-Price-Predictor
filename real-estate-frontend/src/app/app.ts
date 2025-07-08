import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PricePredictionComponent } from './components/price-prediction/price-prediction.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, PricePredictionComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class AppComponent {
  title = 'real-estate-frontend';
}
