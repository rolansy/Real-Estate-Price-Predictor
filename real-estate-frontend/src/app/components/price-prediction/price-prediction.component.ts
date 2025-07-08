import { Component, OnInit, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PricePredictionService } from '../../services/price-prediction.service';

@Component({
  selector: 'app-price-prediction',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './price-prediction.component.html',
  styleUrls: ['./price-prediction.component.css']
})
export class PricePredictionComponent implements OnInit {
  locations: string[] = [];
  selectedLocation: string = '';
  totalSqft: number = 1000;
  bhk: number = 2;
  bath: number = 2;
  estimatedPrice: number | null = null;
  loading: boolean = false;
  error: string = '';

  constructor(
    private priceService: PricePredictionService,
    private ngZone: NgZone,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.loadLocations();
  }

  loadLocations(): void {
    this.priceService.getLocations().subscribe({
      next: (data) => {
        this.ngZone.run(() => {
          this.locations = data.locations;
          if (this.locations.length > 0) {
            this.selectedLocation = this.locations[0];
          }
          this.cdr.detectChanges();
          this.cdr.markForCheck();
        });
      },
      error: (error) => {
        this.ngZone.run(() => {
          this.error = 'Failed to load locations';
          this.cdr.detectChanges();
          this.cdr.markForCheck();
        });
      }
    });
  }

  predictPrice(): void {
    if (!this.selectedLocation) {
      this.error = 'Please select a location';
      this.cdr.detectChanges();
      this.cdr.markForCheck();
      return;
    }

    this.loading = true;
    this.error = '';
    this.estimatedPrice = null;
    this.cdr.detectChanges();
    this.cdr.markForCheck();

    this.priceService.predictPrice(this.totalSqft, this.selectedLocation, this.bhk, this.bath).subscribe({
      next: (data) => {
        this.ngZone.run(() => {
          this.estimatedPrice = data.estimated_price;
          this.loading = false;
          this.cdr.detectChanges();
          this.cdr.markForCheck();
        });
      },
      error: (error) => {
        this.ngZone.run(() => {
          this.error = 'Failed to predict price';
          this.loading = false;
          this.cdr.detectChanges();
          this.cdr.markForCheck();
        });
      }
    });
  }

  onBhkChange(value: number): void {
    this.bhk = value;
    this.cdr.detectChanges();
    this.cdr.markForCheck();
  }

  onBathChange(value: number): void {
    this.bath = value;
    this.cdr.detectChanges();
    this.cdr.markForCheck();
  }
} 